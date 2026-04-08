import importlib
import os
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from bonito.data import DataSettings, ComputeSettings, ModelSetup


VALIDATION_SAMPLE_KEY_NAMESPACE = 1 << 62


def _shape_list(array):
    return [int(dim) for dim in array.shape]


def _build_dataset_config(chunks, targets, lengths, mod_targets):
    return {
        "dataset": {
            "format": "numpy",
            "num_samples": int(lengths.shape[0]),
            "chunk_shape": _shape_list(chunks),
            "reference_shape": _shape_list(targets),
            "reference_lengths_shape": _shape_list(lengths),
            "mod_targets_shape": _shape_list(mod_targets),
        }
    }


def _require_numpy_files(directory, filenames):
    missing = [name for name in filenames if not os.path.exists(os.path.join(directory, name))]
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(f"Missing required dataset files in {directory}: {names}")


def _validate_mod_arrays(chunks, targets, lengths, mod_targets, sample_keys, directory):
    if chunks.ndim != 2:
        raise ValueError(f"{directory}: chunks.npy must have shape [N, chunk_length], got {tuple(chunks.shape)}")
    if targets.ndim != 2:
        raise ValueError(f"{directory}: references.npy must have shape [N, max_target_len], got {tuple(targets.shape)}")
    if lengths.ndim != 1:
        raise ValueError(f"{directory}: reference_lengths.npy must have shape [N], got {tuple(lengths.shape)}")
    if mod_targets.ndim != 2:
        raise ValueError(f"{directory}: mod_targets.npy must have shape [N, max_target_len], got {tuple(mod_targets.shape)}")
    if sample_keys.ndim != 1:
        raise ValueError(f"{directory}: sample_keys must have shape [N], got {tuple(sample_keys.shape)}")

    num_samples = chunks.shape[0]
    if targets.shape[0] != num_samples or lengths.shape[0] != num_samples or mod_targets.shape[0] != num_samples or sample_keys.shape[0] != num_samples:
        raise ValueError(
            f"{directory}: dataset files must share the same first dimension N, got "
            f"chunks={chunks.shape[0]}, references={targets.shape[0]}, reference_lengths={lengths.shape[0]}, "
            f"mod_targets={mod_targets.shape[0]}, sample_keys={sample_keys.shape[0]}"
        )

    if lengths.size and int(lengths.max()) > targets.shape[1]:
        raise ValueError(
            f"{directory}: max(reference_lengths.npy)={int(lengths.max())} exceeds references.npy width={targets.shape[1]}"
        )
    if lengths.size and int(lengths.max()) > mod_targets.shape[1]:
        raise ValueError(
            f"{directory}: max(reference_lengths.npy)={int(lengths.max())} exceeds mod_targets.npy width={mod_targets.shape[1]}"
        )


class TrainModChunkDataSet:
    def __init__(self, chunks, targets, lengths, mod_targets, sample_keys):
        self.chunks = np.expand_dims(chunks, axis=1)
        self.targets = targets
        self.lengths = lengths
        self.mod_targets = mod_targets
        self.sample_keys = sample_keys
        self.dataset_config = _build_dataset_config(chunks, targets, lengths, mod_targets)

    def __getitem__(self, i):
        return (
            self.chunks[i].astype(np.float32),
            self.targets[i].astype(np.int64),
            self.lengths[i].astype(np.int64),
            self.mod_targets[i].astype(np.int64),
            self.sample_keys[i].astype(np.int64),
        )

    def __len__(self):
        return len(self.lengths)


def load_script(directory, name="dataset", suffix=".py", **kwargs):
    directory = Path(directory)
    filepath = (directory / name).with_suffix(suffix)
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loader = module.Loader(**kwargs)
    return loader.train_loader_kwargs(**kwargs), loader.valid_loader_kwargs(**kwargs)


def load_numpy_mod_datasets_with_keys(limit=None, directory=None, namespace=0):
    chunks = np.load(os.path.join(directory, "chunks.npy"), mmap_mode='r')
    targets = np.load(os.path.join(directory, "references.npy"), mmap_mode='r')
    lengths = np.load(os.path.join(directory, "reference_lengths.npy"), mmap_mode='r')
    mod_targets = np.load(os.path.join(directory, "mod_targets.npy"), mmap_mode='r')

    indices_path = os.path.join(directory, "indices.npy")
    if os.path.exists(indices_path):
        sample_keys = np.load(indices_path, mmap_mode='r').astype(np.int64)
        sample_keys = sample_keys[sample_keys < lengths.shape[0]]
        if limit:
            sample_keys = sample_keys[:limit]
        return (
            chunks[sample_keys, :],
            targets[sample_keys, :],
            lengths[sample_keys],
            mod_targets[sample_keys, :],
            np.asarray(sample_keys + np.int64(namespace), dtype=np.int64),
        )

    sample_keys = np.arange(lengths.shape[0], dtype=np.int64)
    if limit:
        chunks = chunks[:limit]
        targets = targets[:limit]
        lengths = lengths[:limit]
        mod_targets = mod_targets[:limit]
        sample_keys = sample_keys[:limit]

    sample_keys = np.asarray(sample_keys + np.int64(namespace), dtype=np.int64)
    _validate_mod_arrays(chunks, targets, lengths, mod_targets, sample_keys, directory)
    return (
        np.array(chunks),
        np.array(targets),
        np.array(lengths),
        np.array(mod_targets),
        sample_keys,
    )


def load_numpy_train_mod(limit, directory, valid_chunks):
    train_data = load_numpy_mod_datasets_with_keys(limit=limit, directory=directory, namespace=0)
    if os.path.exists(os.path.join(directory, "validation")):
        valid_data = load_numpy_mod_datasets_with_keys(
            limit=valid_chunks,
            directory=os.path.join(directory, "validation"),
            namespace=VALIDATION_SAMPLE_KEY_NAMESPACE,
        )
    else:
        print("[validation set not found: splitting training set]")
        split = len(train_data[0]) - valid_chunks
        valid_data = [x[split:] for x in train_data]
        train_data = [x[:split] for x in train_data]
        valid_data[-1] = np.asarray(valid_data[-1] + np.int64(VALIDATION_SAMPLE_KEY_NAMESPACE), dtype=np.int64)

    train_loader_kwargs = {"dataset": TrainModChunkDataSet(*train_data), "shuffle": True}
    valid_loader_kwargs = {"dataset": TrainModChunkDataSet(*valid_data), "shuffle": False}
    return train_loader_kwargs, valid_loader_kwargs


def load_train_mod_data(data: DataSettings, model_setup: ModelSetup, compute_settings: ComputeSettings):
    try:
        if (Path(data.training_data) / "chunks.npy").exists():
            print(f"[loading data] - chunks from {data.training_data}")
            _require_numpy_files(
                data.training_data,
                ("chunks.npy", "references.npy", "reference_lengths.npy", "mod_targets.npy"),
            )
            train_loader_kwargs, valid_loader_kwargs = load_numpy_train_mod(
                data.num_train_chunks,
                data.training_data,
                valid_chunks=data.num_valid_chunks,
            )
        elif (Path(data.training_data) / "dataset.py").exists():
            print(f"[loading data] - dynamically from {data.training_data}/dataset.py")
            train_loader_kwargs, valid_loader_kwargs = load_script(
                data.training_data,
                chunks=data.num_train_chunks,
                valid_chunks=data.num_valid_chunks,
                log_dir=data.output_dir,
                n_pre_context_bases=model_setup.n_pre_context_bases,
                n_post_context_bases=model_setup.n_post_context_bases,
                standardisation=model_setup.standardisation,
                seed=compute_settings.seed,
                batch_size=compute_settings.batch_size,
                num_workers=compute_settings.num_workers,
            )
        else:
            raise FileNotFoundError(f"No suitable training data found at: {data.training_data}")
    except Exception as e:
        raise IOError(f"Failed to load input data from {data.training_data}") from e

    default_settings = {
        "batch_size": compute_settings.batch_size,
        "num_workers": compute_settings.num_workers,
        "pin_memory": compute_settings.pin_memory,
    }
    train_loader = DataLoader(**{**default_settings, **train_loader_kwargs})
    valid_loader = DataLoader(**{**default_settings, **valid_loader_kwargs})
    return train_loader, valid_loader
