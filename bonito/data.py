import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from torch.utils.data import DataLoader


@dataclass
class DataSettings:
    training_data: Path
    num_train_chunks: int
    num_valid_chunks: int
    output_dir: Path

@dataclass
class ComputeSettings:
    batch_size: int
    num_workers: int
    seed: int
    pin_memory: bool = True

@dataclass
class ModelSetup:
    n_pre_context_bases: int
    n_post_context_bases: int
    standardisation: Dict


def _shape_list(array):
    return [int(dim) for dim in array.shape]


def _build_dataset_config(chunks, targets, lengths, mod_targets=None):
    dataset = {
        "format": "numpy",
        "num_samples": int(lengths.shape[0]),
        "chunk_shape": _shape_list(chunks),
        "reference_shape": _shape_list(targets),
        "reference_lengths_shape": _shape_list(lengths),
    }
    if mod_targets is not None:
        dataset["mod_targets_shape"] = _shape_list(mod_targets)
    return {"dataset": dataset}


def _require_numpy_files(directory, filenames):
    missing = [name for name in filenames if not os.path.exists(os.path.join(directory, name))]
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(f"Missing required dataset files in {directory}: {names}")


def _validate_core_arrays(chunks, targets, lengths, directory):
    if chunks.ndim != 2:
        raise ValueError(f"{directory}: chunks.npy must have shape [N, chunk_length], got {tuple(chunks.shape)}")
    if targets.ndim != 2:
        raise ValueError(f"{directory}: references.npy must have shape [N, max_target_len], got {tuple(targets.shape)}")
    if lengths.ndim != 1:
        raise ValueError(f"{directory}: reference_lengths.npy must have shape [N], got {tuple(lengths.shape)}")

    num_samples = chunks.shape[0]
    if targets.shape[0] != num_samples or lengths.shape[0] != num_samples:
        raise ValueError(
            f"{directory}: dataset files must share the same first dimension N, got "
            f"chunks={chunks.shape[0]}, references={targets.shape[0]}, reference_lengths={lengths.shape[0]}"
        )

    if lengths.size and int(lengths.max()) > targets.shape[1]:
        raise ValueError(
            f"{directory}: max(reference_lengths.npy)={int(lengths.max())} exceeds references.npy width={targets.shape[1]}"
        )


def _validate_mod_arrays(chunks, targets, lengths, mod_targets, directory):
    _validate_core_arrays(chunks, targets, lengths, directory)

    if mod_targets.ndim != 2:
        raise ValueError(f"{directory}: mod_targets.npy must have shape [N, max_target_len], got {tuple(mod_targets.shape)}")
    if mod_targets.shape[0] != chunks.shape[0]:
        raise ValueError(
            f"{directory}: mod_targets.npy must share the same first dimension N as chunks.npy, got "
            f"chunks={chunks.shape[0]}, mod_targets={mod_targets.shape[0]}"
        )
    if lengths.size and int(lengths.max()) > mod_targets.shape[1]:
        raise ValueError(
            f"{directory}: max(reference_lengths.npy)={int(lengths.max())} exceeds mod_targets.npy width={mod_targets.shape[1]}"
        )


def load_data(data, model_setup, compute_settings):
    try:
        if (Path(data.training_data) / "chunks.npy").exists():
            print(f"[loading data] - chunks from {data.training_data}")
            _require_numpy_files(data.training_data, ("chunks.npy", "references.npy", "reference_lengths.npy"))
            train_loader_kwargs, valid_loader_kwargs = load_numpy(
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

    # Allow options from the train/valid_loader to override the default_kwargs
    train_loader = DataLoader(**{**default_settings, **train_loader_kwargs})
    valid_loader = DataLoader(**{**default_settings, **valid_loader_kwargs})
    return train_loader, valid_loader


class ChunkDataSet:
    def __init__(self, chunks, targets, lengths):
        self.chunks = np.expand_dims(chunks, axis=1)
        self.targets = targets
        self.lengths = lengths
        self.dataset_config = _build_dataset_config(chunks, targets, lengths)

    def __getitem__(self, i):
        return (
            self.chunks[i].astype(np.float32),
            self.targets[i].astype(np.int64),
            self.lengths[i].astype(np.int64),
        )

    def __len__(self):
        return len(self.lengths)


class ModChunkDataSet:
    def __init__(self, chunks, targets, lengths, mod_targets):
        self.chunks = np.expand_dims(chunks, axis=1)
        self.targets = targets
        self.lengths = lengths
        self.mod_targets = mod_targets
        self.dataset_config = _build_dataset_config(chunks, targets, lengths, mod_targets=mod_targets)

    def __getitem__(self, i):
        return (
            self.chunks[i].astype(np.float32),
            self.targets[i].astype(np.int64),
            self.lengths[i].astype(np.int64),
            self.mod_targets[i].astype(np.int64),
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


def load_numpy(limit, directory, valid_chunks):
    """
    Returns training and validation DataLoaders for data in directory.
    """
    train_data = load_numpy_datasets(limit=limit, directory=directory)
    if os.path.exists(os.path.join(directory, 'validation')):
        valid_data = load_numpy_datasets(limit=valid_chunks,
            directory=os.path.join(directory, 'validation')
        )
    else:
        print("[validation set not found: splitting training set]")
        split = len(train_data[0]) - valid_chunks
        valid_data = [x[split:] for x in train_data]
        train_data = [x[:split] for x in train_data]

    train_loader_kwargs = {"dataset": ChunkDataSet(*train_data), "shuffle": True}
    valid_loader_kwargs = {"dataset": ChunkDataSet(*valid_data), "shuffle": False}
    return train_loader_kwargs, valid_loader_kwargs


def load_mod_data(data, model_setup, compute_settings):
    try:
        if (Path(data.training_data) / "chunks.npy").exists():
            print(f"[loading data] - chunks from {data.training_data}")
            _require_numpy_files(
                data.training_data,
                ("chunks.npy", "references.npy", "reference_lengths.npy", "mod_targets.npy"),
            )
            train_loader_kwargs, valid_loader_kwargs = load_numpy_mod(
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


def load_numpy_datasets(limit=None, directory=None):
    """
    Returns numpy chunks, targets and lengths arrays.
    """
    chunks = np.load(os.path.join(directory, "chunks.npy"), mmap_mode='r')
    targets = np.load(os.path.join(directory, "references.npy"), mmap_mode='r')
    lengths = np.load(os.path.join(directory, "reference_lengths.npy"), mmap_mode='r')

    indices = os.path.join(directory, "indices.npy")

    if os.path.exists(indices):
        idx = np.load(indices, mmap_mode='r')
        idx = idx[idx < lengths.shape[0]]
        if limit:
            idx = idx[:limit]
        return chunks[idx, :], targets[idx, :], lengths[idx]

    if limit:
        chunks = chunks[:limit]
        targets = targets[:limit]
        lengths = lengths[:limit]

    _validate_core_arrays(chunks, targets, lengths, directory)
    return np.array(chunks), np.array(targets), np.array(lengths)


def load_numpy_mod(limit, directory, valid_chunks):
    train_data = load_numpy_mod_datasets(limit=limit, directory=directory)
    if os.path.exists(os.path.join(directory, 'validation')):
        valid_data = load_numpy_mod_datasets(limit=valid_chunks,
            directory=os.path.join(directory, 'validation')
        )
    else:
        print("[validation set not found: splitting training set]")
        split = len(train_data[0]) - valid_chunks
        valid_data = [x[split:] for x in train_data]
        train_data = [x[:split] for x in train_data]

    train_loader_kwargs = {"dataset": ModChunkDataSet(*train_data), "shuffle": True}
    valid_loader_kwargs = {"dataset": ModChunkDataSet(*valid_data), "shuffle": False}
    return train_loader_kwargs, valid_loader_kwargs


def load_numpy_mod_datasets(limit=None, directory=None):
    chunks = np.load(os.path.join(directory, "chunks.npy"), mmap_mode='r')
    targets = np.load(os.path.join(directory, "references.npy"), mmap_mode='r')
    lengths = np.load(os.path.join(directory, "reference_lengths.npy"), mmap_mode='r')
    mod_targets = np.load(os.path.join(directory, "mod_targets.npy"), mmap_mode='r')

    indices = os.path.join(directory, "indices.npy")

    if os.path.exists(indices):
        idx = np.load(indices, mmap_mode='r')
        idx = idx[idx < lengths.shape[0]]
        if limit:
            idx = idx[:limit]
        return chunks[idx, :], targets[idx, :], lengths[idx], mod_targets[idx, :]

    if limit:
        chunks = chunks[:limit]
        targets = targets[:limit]
        lengths = lengths[:limit]
        mod_targets = mod_targets[:limit]

    _validate_mod_arrays(chunks, targets, lengths, mod_targets, directory)
    return np.array(chunks), np.array(targets), np.array(lengths), np.array(mod_targets)

