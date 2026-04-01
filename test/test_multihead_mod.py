import unittest
from unittest import mock

import numpy as np

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
    from bonito.transformer.multihead_model import MultiHeadModel


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

        with mock.patch.object(model, "_decode_paths", return_value=path):
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
