# Bonito

[![PyPI version](https://badge.fury.io/py/ont-bonito.svg)](https://badge.fury.io/py/ont-bonito)
[![py310](https://img.shields.io/badge/python-3.10-brightgreen.svg)](https://img.shields.io/badge/python-3.10-brightgreen.svg)
[![py311](https://img.shields.io/badge/python-3.11-brightgreen.svg)](https://img.shields.io/badge/python-3.11-brightgreen.svg)
[![py312](https://img.shields.io/badge/python-3.12-brightgreen.svg)](https://img.shields.io/badge/python-3.12-brightgreen.svg)
[![py313](https://img.shields.io/badge/python-3.13-brightgreen.svg)](https://img.shields.io/badge/python-3.13-brightgreen.svg)
[![py314](https://img.shields.io/badge/python-3.14-brightgreen.svg)](https://img.shields.io/badge/python-3.14-brightgreen.svg)
[![cu128](https://img.shields.io/badge/cuda-12.8-blue.svg)](https://img.shields.io/badge/cuda-12.8-blue.svg)
[![cu130](https://img.shields.io/badge/cuda-13.0-blue.svg)](https://img.shields.io/badge/cuda-13.0-blue.svg)

Bonito is an open source research basecaller for Oxford Nanopore reads.

For anything other than basecaller training or method development please use [dorado](https://github.com/nanoporetech/dorado).

```bash
$ pip install --upgrade pip
$ pip install ont-bonito
$ bonito basecaller dna_r10.4.1_e8.2_400bps_hac@v5.0.0 /data/reads > basecalls.bam
```

Bonito supports writing aligned/unaligned `{fastq, sam, bam, cram}`.

```bash
$ bonito basecaller dna_r10.4.1_e8.2_400bps_hac@v5.0.0 --reference reference.mmi /data/reads > basecalls.bam
```

Bonito will download and cache the basecalling model automatically on first use but all models can be downloaded with -

``` bash
$ bonito download --models --show  # show all available models
$ bonito download --models         # download all available models
```

## Modified Bases

For modified-base calling with ont-supported mods please use [dorado](https://github.com/nanoporetech/dorado)
For development of modified base calling models please see [remora](https://github.com/nanoporetech/remora).

## Training your own model

For detailed information on the training process, please see the [Training Documentation](documentation/training.md).

## Developer Quickstart

```bash
$ git clone https://github.com/nanoporetech/bonito.git  # or fork first and clone that
$ cd bonito
$ python3 -m venv venv3
$ source venv3/bin/activate
(venv3) $ pip install --upgrade pip
(venv3) $ pip install -e .[cu128] --extra-index-url https://download.pytorch.org/whl/cu128
```

## Using `uv` for local development

If you prefer [`uv`](https://github.com/astral-sh/uv) to manage the virtual
environment and perform editable installs without touching your system
Python, clone the repo and run:

```bash
$ uv venv --python 3.11 .venv         # or 3.10/3.12 as needed
$ source .venv/bin/activate
(.venv) $ uv pip install --upgrade pip
(.venv) $ uv pip install -e .[cu128] --extra-index-url https://download.pytorch.org/whl/cu128
```

Notes:

- The editable install (`-e`) keeps changes in your working copy immediately
  importable. Use `[cu130]` (or omit the CUDA extra) if that matches your
  system better.
- If you do **not** want to install the package name, set `PYTHONPATH=$(pwd)`
  after `uv pip install -r requirements.txt` and invoke commands with
  `python -m bonito <subcommand>` instead of `bonito <subcommand>`.
- `uv` respects the same dependency extras as `pip`, so any existing
  instructions that reference `pip install -e .[...]` can be swapped directly
  with `uv pip install -e .[...]` when the environment is activated.

The `ont-bonito[cu128]` and `ont-bonito[cu130]` optional dependencies can be used, along
with the corresponding `--extra-index-url`, to ensure the PyTorch package matches the
local CUDA setup.

## Interface

 - `bonito view` - view a model architecture for a given `.toml` file and the number of parameters in the network.
 - `bonito train` - train a bonito model.
 - `bonito evaluate` - evaluate a model performance.
 - `bonito download` - download pretrained models and training datasets.
 - `bonito basecaller` - basecaller *(`.pod5` -> `.bam`)*.
 - `bonito model-info` - print a torchinfo summary for a cached or downloaded basecaller model.

## Basecaller model summary

Use the `model-info` subcommand to download (if needed) and inspect a basecaller with
[`torchinfo`](https://github.com/TylerYep/torchinfo). The example below targets CPU and
uses a single sample with a 4000-length signal:

```bash
$ bonito model-info dna_r10.4.1_e8.2_400bps_hac@v5.2.0 --device cpu --signal-length 4000 --batch-size 1
```

Pass `--return-all-heads` when summarising multi-head models (e.g. custom configs that
include modification heads).

### References

 - [Sequence Modeling With CTC](https://distill.pub/2017/ctc/)
 - [Quartznet: Deep Automatic Speech Recognition With 1D Time-Channel Separable Convolutions](https://arxiv.org/pdf/1910.10261.pdf)
 - [Pair consensus decoding improves accuracy of neural network basecallers for nanopore sequencing](https://www.biorxiv.org/content/10.1101/2020.02.25.956771v1.full.pdf)

### Licence and Copyright
(c) 2019 Oxford Nanopore Technologies Ltd.

Bonito is distributed under the terms of the Oxford Nanopore
Technologies, Ltd.  Public License, v. 1.0.  If a copy of the License
was not distributed with this file, You can obtain one at
http://nanoporetech.com

### Research Release

Research releases are provided as technology demonstrators to provide early access to features or stimulate Community development of tools. Support for this software will be minimal and is only provided directly by the developers. Feature requests, improvements, and discussions are welcome and can be implemented by forking and pull requests. However much as we would like to rectify every issue and piece of feedback users may have, the developers may have limited resource for support of this software. Research releases may be unstable and subject to rapid iteration by Oxford Nanopore Technologies.

### Citation

```
@software{bonito,
  title = {Bonito: A PyTorch Basecaller for Oxford Nanopore Reads},
  author = {{Chris Seymour, Oxford Nanopore Technologies Ltd.}},
  year = {2019},
  url = {https://github.com/nanoporetech/bonito},
  note = {Oxford Nanopore Technologies, Ltd. Public License, v. 1.0},
  abstract = {Bonito is an open source research basecaller for Oxford Nanopore reads. It provides a flexible platform for training and developing basecalling models using PyTorch.}
}
```