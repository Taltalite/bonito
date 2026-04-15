import unittest
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import toml

from gen_data.make_mod_targets_m6a import (
    MODE_CANONICAL,
    MODE_FULL_MOD,
    NON_A_POLICY_CANONICAL,
    NON_A_POLICY_IGNORE,
    build_mod_targets,
)

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from bonito.cli.train_mod import merge_pretrained_runtime_config, validate_pretrained_runtime_config
    from bonito.transformer import multihead_basecall
    from bonito.transformer.multihead_model import MultiHeadModel
    from bonito.util import load_model


@unittest.skipUnless(torch is not None, "torch is required for model tests")
class TestMultiHeadModModel(unittest.TestCase):
    def _config(self, labels=None):
        return {
            "model": {
                "package": "bonito.transformer.multihead_model",
                "d_model": 16,
                "nhead": 2,
                "dim_feedforward": 32,
                "num_layers": 1,
                "kernel_size": 3,
                "stride": 2,
                "mod_loss_weight": 1.0,
                "mod_target_projection": "viterbi_edlib_equal",
                "mod_decode_projection": "viterbi_path",
                "mod_bases": ["A", "C", "G", "T"],
                "mod_global_labels": [
                    "canonical_A",
                    "canonical_C",
                    "canonical_G",
                    "canonical_T",
                    "m6A",
                ],
                "mod_trunk_dim": 8,
                "mod_trunk_kernel_size": 3,
                "mod_trunk_depth": 1,
                "mod_head_dropout": 0.0,
                "mod_head_defs": {
                    "A": ["canonical_A", "m6A"],
                    "C": ["canonical_C"],
                    "G": ["canonical_G"],
                    "T": ["canonical_T"],
                },
                "base_slot_aliases": {
                    "T": ["T", "U"],
                },
            },
            "input": {
                "features": 1,
                "n_pre_post_context_bases": [0, 0],
            },
            "labels": {
                "labels": labels or ["", "A", "C", "G", "T"],
            },
            "global_norm": {
                "state_len": 4,
            },
        }

    def test_forward_returns_four_mod_heads(self):
        model = MultiHeadModel(self._config())
        outputs = model(torch.randn(1, 1, 16))

        self.assertIn("base_scores", outputs)
        self.assertIn("mod_logits_by_base", outputs)
        self.assertEqual(set(outputs["mod_logits_by_base"].keys()), {"A", "C", "G", "T"})
        self.assertEqual(outputs["mod_logits_by_base"]["A"].shape[-1], 2)
        self.assertEqual(outputs["mod_logits_by_base"]["C"].shape[-1], 1)
        self.assertEqual(outputs["mod_logits_by_base"]["G"].shape[-1], 1)
        self.assertEqual(outputs["mod_logits_by_base"]["T"].shape[-1], 1)

    def test_alignment_routes_global_targets_to_per_base_heads(self):
        model = MultiHeadModel(self._config())
        outputs = model(torch.randn(1, 1, 16))
        time_steps = outputs["base_scores"].shape[0]
        path = torch.zeros((1, time_steps), dtype=torch.int16)
        path[0, :4] = torch.tensor([1, 2, 3, 4], dtype=torch.int16)

        targets = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
        lengths = torch.tensor([4], dtype=torch.int64)
        mod_targets = torch.tensor([[4, 1, 2, 3]], dtype=torch.int64)

        with mock.patch.object(model, "_decode_paths", return_value=path), \
             mock.patch.object(model.seqdist, "ctc_loss", return_value=torch.tensor(0.5)):
            projection = model.align_predictions_to_targets(outputs, targets, lengths, mod_targets)
            losses = model.loss(outputs, targets, lengths, mod_targets)

        self.assertEqual(projection["per_head"]["A"]["flat_targets"].tolist(), [1])
        self.assertEqual(projection["per_head"]["C"]["flat_targets"].tolist(), [0])
        self.assertEqual(projection["per_head"]["G"]["flat_targets"].tolist(), [0])
        self.assertEqual(projection["per_head"]["T"]["flat_targets"].tolist(), [0])
        self.assertIn("mod_loss", losses)
        self.assertTrue(torch.isfinite(losses["mod_loss"]))

    def test_predict_mods_uses_t_u_slot_alias(self):
        model = MultiHeadModel(self._config(labels=["", "A", "C", "G", "U"]))
        outputs = model(torch.randn(1, 1, 16))
        time_steps = outputs["base_scores"].shape[0]
        path = torch.zeros((1, time_steps), dtype=torch.int16)
        path[0, :4] = torch.tensor([1, 2, 3, 4], dtype=torch.int16)

        with mock.patch.object(model, "_decode_paths", return_value=path):
            preds = model.predict_mods(outputs)

        self.assertEqual(preds[0]["sites"][-1]["base_label"], "U")
        self.assertEqual(preds[0]["sites"][-1]["head_name"], "T/U slot")
        self.assertEqual(preds[0]["sites"][-1]["global_pred_label"], "canonical_T")

    def test_decoding_is_cached_per_outputs_object(self):
        model = MultiHeadModel(self._config())
        outputs = model(torch.randn(1, 1, 16))
        time_steps = outputs["base_scores"].shape[0]
        path = torch.zeros((1, time_steps), dtype=torch.int16)
        path[0, :4] = torch.tensor([1, 2, 3, 4], dtype=torch.int16)

        targets = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
        lengths = torch.tensor([4], dtype=torch.int64)
        mod_targets = torch.tensor([[4, 1, 2, 3]], dtype=torch.int64)

        with mock.patch.object(model, "_decode_paths", return_value=path) as decode_paths:
            decoded = model.decode_batch(outputs)
            preds = model.predict_mods(outputs)
            projection = model.align_predictions_to_targets(outputs, targets, lengths, mod_targets)

        self.assertEqual(decoded, ["ACGT"])
        self.assertEqual(preds[0]["sequence"], "ACGT")
        self.assertEqual(projection["sample_records"][0]["aligned_equal_bases"], 4)
        decode_paths.assert_called_once()

    def test_standalone_alignment_cache_reuses_projection_across_outputs(self):
        config = self._config()
        config["training"] = {
            "mode": "standalone_mod_head",
            "pretrained_basecaller": "dummy",
        }
        model = MultiHeadModel(config)
        outputs_1 = model(torch.randn(1, 1, 16), torch.tensor([7], dtype=torch.int64))
        outputs_2 = model(torch.randn(1, 1, 16), torch.tensor([7], dtype=torch.int64))
        time_steps = outputs_1["base_scores"].shape[0]
        path = torch.zeros((1, time_steps), dtype=torch.int16)
        path[0, :4] = torch.tensor([1, 2, 3, 4], dtype=torch.int16)

        targets = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
        lengths = torch.tensor([4], dtype=torch.int64)
        mod_targets = torch.tensor([[4, 1, 2, 3]], dtype=torch.int64)

        with mock.patch.object(model, "_decode_paths", return_value=path) as decode_paths, \
             mock.patch("bonito.transformer.multihead_model.edlib.align", return_value={"cigar": "4="}) as edlib_align:
            losses_1 = model.loss(outputs_1, targets, lengths, mod_targets)
            losses_2 = model.loss(outputs_2, targets, lengths, mod_targets)

        decode_paths.assert_called_once()
        edlib_align.assert_called_once()
        self.assertTrue(torch.isfinite(losses_1["mod_loss"]))
        self.assertTrue(torch.isfinite(losses_2["mod_loss"]))

    def test_alignment_cache_is_bypassed_for_site_records(self):
        config = self._config()
        config["training"] = {
            "mode": "standalone_mod_head",
            "pretrained_basecaller": "dummy",
        }
        model = MultiHeadModel(config)
        outputs_1 = model(torch.randn(1, 1, 16), torch.tensor([9], dtype=torch.int64))
        outputs_2 = model(torch.randn(1, 1, 16), torch.tensor([9], dtype=torch.int64))
        time_steps = outputs_1["base_scores"].shape[0]
        path = torch.zeros((1, time_steps), dtype=torch.int16)
        path[0, :4] = torch.tensor([1, 2, 3, 4], dtype=torch.int16)

        targets = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
        lengths = torch.tensor([4], dtype=torch.int64)
        mod_targets = torch.tensor([[4, 1, 2, 3]], dtype=torch.int64)

        with mock.patch.object(model, "_decode_paths", return_value=path) as decode_paths, \
             mock.patch("bonito.transformer.multihead_model.edlib.align", return_value={"cigar": "4="}) as edlib_align:
            model.align_predictions_to_targets(outputs_1, targets, lengths, mod_targets, include_site_records=True)
            model.align_predictions_to_targets(outputs_2, targets, lengths, mod_targets, include_site_records=True)

        self.assertEqual(decode_paths.call_count, 2)
        self.assertEqual(edlib_align.call_count, 2)

    def test_standalone_mode_freezes_basecaller_and_keeps_it_in_eval(self):
        config = self._config()
        config["training"] = {
            "mode": "standalone_mod_head",
            "pretrained_basecaller": "dummy",
        }
        model = MultiHeadModel(config)
        model.train()

        self.assertFalse(any(param.requires_grad for param in model.conv.parameters()))
        self.assertFalse(any(param.requires_grad for layer in model.encoder_layers for param in layer.parameters()))
        self.assertFalse(any(param.requires_grad for param in model.crf.parameters()))
        self.assertTrue(any(param.requires_grad for param in model.mod_input_proj.parameters()))
        self.assertTrue(any(param.requires_grad for param in model.mod_heads.parameters()))
        self.assertFalse(model.conv.training)
        self.assertFalse(model.encoder_layers.training)
        self.assertFalse(model.crf.training)
        self.assertTrue(model.mod_input_proj.training)
        self.assertTrue(model.mod_heads.training)

    def test_standalone_checkpoint_only_contains_mod_branch(self):
        config = self._config()
        config["training"] = {
            "mode": "standalone_mod_head",
            "pretrained_basecaller": "dummy",
        }
        model = MultiHeadModel(config)
        checkpoint = model.checkpoint_state_dict()

        self.assertTrue(checkpoint)
        self.assertTrue(all(name.startswith(("mod_input_proj.", "mod_trunk.", "mod_heads.")) for name in checkpoint))
        self.assertFalse(any(name.startswith(("conv.", "encoder_layers.", "crf.")) for name in checkpoint))

    def test_standalone_total_loss_optimizes_only_mod_loss(self):
        config = self._config()
        config["training"] = {
            "mode": "standalone_mod_head",
            "pretrained_basecaller": "dummy",
        }
        model = MultiHeadModel(config)
        outputs = model(torch.randn(1, 1, 16))
        time_steps = outputs["base_scores"].shape[0]
        path = torch.zeros((1, time_steps), dtype=torch.int16)
        path[0, :4] = torch.tensor([1, 2, 3, 4], dtype=torch.int16)

        targets = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
        lengths = torch.tensor([4], dtype=torch.int64)
        mod_targets = torch.tensor([[4, 1, 2, 3]], dtype=torch.int64)

        with mock.patch.object(model, "_decode_paths", return_value=path), \
             mock.patch.object(model.seqdist, "ctc_loss", return_value=torch.tensor(0.5)) as ctc_loss:
            losses = model.loss(outputs, targets, lengths, mod_targets)

        ctc_loss.assert_not_called()
        self.assertTrue(torch.equal(losses["loss"], torch.zeros_like(losses["loss"])))
        self.assertTrue(torch.equal(losses["base_loss"], torch.zeros_like(losses["base_loss"])))
        self.assertTrue(torch.allclose(losses["total_loss"], losses["mod_loss"]))

    def test_non_standalone_loss_still_computes_base_loss(self):
        model = MultiHeadModel(self._config())
        outputs = model(torch.randn(1, 1, 16))
        time_steps = outputs["base_scores"].shape[0]
        path = torch.zeros((1, time_steps), dtype=torch.int16)
        path[0, :4] = torch.tensor([1, 2, 3, 4], dtype=torch.int16)

        targets = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
        lengths = torch.tensor([4], dtype=torch.int64)
        mod_targets = torch.tensor([[4, 1, 2, 3]], dtype=torch.int64)

        with mock.patch.object(model, "_decode_paths", return_value=path), \
             mock.patch.object(model.seqdist, "ctc_loss", return_value=torch.tensor(0.5)) as ctc_loss:
            losses = model.loss(outputs, targets, lengths, mod_targets)

        ctc_loss.assert_called_once()
        self.assertTrue(torch.allclose(losses["base_loss"], torch.tensor(0.5)))

    def test_standalone_load_model_reconstructs_frozen_basecaller(self):
        torch.manual_seed(7)
        base_config = self._config()
        official_model = MultiHeadModel(base_config)
        base_only_state = {
            name: tensor.clone()
            for name, tensor in official_model.state_dict().items()
            if not name.startswith(("mod_input_proj.", "mod_trunk.", "mod_heads."))
        }

        standalone_config = self._config()
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            official_dir = tmpdir / "official"
            official_dir.mkdir()
            with (official_dir / "config.toml").open("w") as fh:
                toml.dump(base_config, fh)
            torch.save(base_only_state, official_dir / "weights_1.tar")

            standalone_config["training"] = {
                "mode": "standalone_mod_head",
                "pretrained_basecaller": str(official_dir),
            }
            standalone_model = MultiHeadModel(standalone_config)
            mod_state = standalone_model.checkpoint_state_dict()
            for tensor in mod_state.values():
                tensor.add_(0.5)

            run_dir = tmpdir / "run"
            run_dir.mkdir()
            with (run_dir / "config.toml").open("w") as fh:
                toml.dump(standalone_config, fh)
            torch.save(mod_state, run_dir / "weights_1.tar")

            loaded = load_model(str(run_dir), device="cpu", half=False, compile=False)
            loaded_state = loaded.state_dict()

            for name, tensor in base_only_state.items():
                self.assertTrue(torch.allclose(loaded_state[name], tensor), name)
            for name, tensor in mod_state.items():
                self.assertTrue(torch.allclose(loaded_state[name], tensor), name)

            signal = torch.randn(1, 1, 16)
            official_model.eval()
            loaded.eval()
            official_outputs = official_model(signal)
            loaded_outputs = loaded(signal)
            self.assertTrue(
                torch.allclose(
                    official_outputs["base_scores"],
                    loaded_outputs["base_scores"],
                    atol=1e-6,
                    rtol=1e-5,
                )
            )


@unittest.skipUnless(torch is not None, "torch is required for model tests")
class TestStandaloneModConfigCompatibility(unittest.TestCase):
    def _standalone_config(self):
        return {
            "model": {
                "package": "bonito.transformer.multihead_model",
                "d_model": 16,
                "nhead": 2,
                "dim_feedforward": 32,
                "num_layers": 1,
                "kernel_size": 3,
                "stride": 2,
                "mod_loss_weight": 1.0,
                "mod_target_projection": "viterbi_edlib_equal",
                "mod_decode_projection": "viterbi_path",
                "mod_bases": ["A", "C", "G", "T"],
                "mod_global_labels": [
                    "canonical_A",
                    "canonical_C",
                    "canonical_G",
                    "canonical_T",
                    "m6A",
                ],
                "mod_trunk_dim": 8,
                "mod_trunk_kernel_size": 3,
                "mod_trunk_depth": 1,
                "mod_head_dropout": 0.0,
                "mod_head_defs": {
                    "A": ["canonical_A", "m6A"],
                    "C": ["canonical_C"],
                    "G": ["canonical_G"],
                    "T": ["canonical_T"],
                },
                "base_slot_aliases": {
                    "T": ["T", "U"],
                },
            },
            "input": {
                "features": 1,
                "n_pre_post_context_bases": [0, 0],
            },
            "labels": {
                "labels": ["", "A", "C", "G", "T"],
            },
            "global_norm": {
                "state_len": 4,
            },
        }

    def _official_pretrained_config(self):
        return {
            "basecaller": {
                "batchsize": 128,
                "chunksize": 12000,
                "overlap": 600,
            },
            "scaling": {
                "strategy": "pa",
            },
            "standardisation": {
                "standardise": 1,
                "mean": 79.17,
                "stdev": 16.93,
            },
            "run_info": {
                "sample_type": "rna",
                "sample_rate": 4000,
            },
            "qscore": {
                "strategy": "trim_polyA",
                "scale": 1.25,
                "bias": 1.5,
            },
            "model": {
                "seqdist": {
                    "state_len": 5,
                    "alphabet": ["N", "A", "C", "G", "T"],
                },
                "encoder": {
                    "conv": {
                        "sublayers": [
                            {
                                "type": "convolution",
                                "insize": 1,
                                "size": 64,
                            },
                        ],
                    },
                    "crf": {
                        "state_len": 5,
                    },
                },
            },
        }

    def test_merge_pretrained_runtime_config_copies_runtime_fields(self):
        config = self._standalone_config()
        pretrained_config = self._official_pretrained_config()

        merged = merge_pretrained_runtime_config(config, pretrained_config)

        self.assertEqual(merged["basecaller"], pretrained_config["basecaller"])
        self.assertEqual(merged["scaling"], pretrained_config["scaling"])
        self.assertEqual(merged["standardisation"], pretrained_config["standardisation"])
        self.assertEqual(merged["run_info"], pretrained_config["run_info"])
        self.assertEqual(merged["qscore"], pretrained_config["qscore"])
        self.assertEqual(merged["labels"]["labels"], ["N", "A", "C", "G", "T"])
        self.assertEqual(merged["input"]["features"], 1)
        self.assertEqual(merged["global_norm"]["state_len"], 5)

    def test_validate_pretrained_runtime_config_rejects_missing_scaling(self):
        config = self._standalone_config()
        pretrained_config = self._official_pretrained_config()
        merged = merge_pretrained_runtime_config(config, pretrained_config)
        merged.pop("scaling")

        with self.assertRaisesRegex(ValueError, "missing scaling"):
            validate_pretrained_runtime_config(merged, pretrained_config)

    def test_validate_pretrained_runtime_config_checks_model_state_len(self):
        config = self._standalone_config()
        pretrained_config = self._official_pretrained_config()
        merged = merge_pretrained_runtime_config(config, pretrained_config)

        class DummySeqdist:
            state_len = 4

        class DummyModel:
            seqdist = DummySeqdist()

        with self.assertRaisesRegex(ValueError, "state_len does not match"):
            validate_pretrained_runtime_config(merged, pretrained_config, model=DummyModel())


@unittest.skipUnless(torch is not None, "torch is required for model tests")
class TestMultiHeadBasecallDecodePath(unittest.TestCase):
    def test_decode_basecall_batch_delegates_to_shared_decode_scores(self):
        model = mock.Mock()
        model.seqdist = object()
        base_scores = mock.Mock()
        base_scores.ndim = 3
        base_scores.device = "cuda:0"

        sentinel = {"sequence": "ACGT"}
        with mock.patch("bonito.transformer.multihead_basecall.decode_scores", return_value=sentinel) as decode_scores:
            result = multihead_basecall._decode_basecall_batch(model, base_scores, reverse=True)

        self.assertIs(result, sentinel)
        decode_scores.assert_called_once_with(base_scores, model.seqdist, reverse=True)


class TestMakeModTargetsM6A(unittest.TestCase):
    def test_full_mod_canonical_policy_uses_global_label_scheme(self):
        references = np.array([[1, 2, 3, 4]], dtype=np.int64)
        lengths = np.array([4], dtype=np.int64)

        mod_targets = build_mod_targets(
            references=references,
            lengths=lengths,
            mode=MODE_FULL_MOD,
            non_a_policy=NON_A_POLICY_CANONICAL,
            ignore_value=-100,
        )

        self.assertEqual(mod_targets.tolist(), [[4, 1, 2, 3]])

    def test_canonical_mode_and_ignore_policy(self):
        references = np.array([[1, 2, 3, 4]], dtype=np.int64)
        lengths = np.array([4], dtype=np.int64)

        mod_targets = build_mod_targets(
            references=references,
            lengths=lengths,
            mode=MODE_CANONICAL,
            non_a_policy=NON_A_POLICY_IGNORE,
            ignore_value=-100,
        )

        self.assertEqual(mod_targets.tolist(), [[0, -100, -100, -100]])
