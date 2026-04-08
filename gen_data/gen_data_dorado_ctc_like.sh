#!/usr/bin/env bash
set -euo pipefail

# Example command template for Dorado-BAM-driven Bonito train_mod datasets.
# Replace paths before use.

# mod dataset
python gen_data/create_dataset_dorado_ctc_like.py \
  --bam-file /data/path/to/mod/basecaller_with_moves.bam \
  --pod5-dir /data/path/to/mod/pod5 \
  --reference-fasta /data/path/to/reference.fa \
  --output-dir /data/path/to/mod_dataset \
  --sample-type rna \
  --chunk-len 12000 \
  --overlap 600 \
  --filter-preset strict \
  --norm-strategy from-bam \
  --workers 8

python gen_data/make_mod_targets_m6a.py \
  --dataset-dir /data/path/to/mod_dataset \
  --mode full-mod \
  --non-a-policy ignore


# canonical dataset
python gen_data/create_dataset_dorado_ctc_like.py \
  --bam-file /data/path/to/canonical/basecaller_with_moves.bam \
  --pod5-dir /data/path/to/canonical/pod5 \
  --reference-fasta /data/path/to/reference.fa \
  --output-dir /data/path/to/canonical_dataset \
  --sample-type rna \
  --chunk-len 12000 \
  --overlap 600 \
  --filter-preset strict \
  --norm-strategy from-bam \
  --workers 8

python gen_data/make_mod_targets_m6a.py \
  --dataset-dir /data/path/to/canonical_dataset \
  --mode canonical \
  --non-a-policy ignore


# merge into one train_mod dataset
python gen_data/merge_mod_datasets.py \
  --full-mod-dir /data/path/to/mod_dataset \
  --canonical-dir /data/path/to/canonical_dataset \
  --output-dir /data/path/to/mix_dataset


# train
bonito train_mod -f /data/path/to/training_model/rna004_m6a_mix_ft \
  --directory /data/path/to/mix_dataset \
  --config /home/lijy/workspace/bonito/bonito/models/configs/multihead_transformer.toml \
  --pretrained /home/lijy/workspace/bonito/bonito/models/rna004_130bps_sup@v5.2.0 \
  --epochs 30 \
  --batch 48 \
  --lr 5e-5 \
  --chunks 300000 \
  --valid-chunks 20000 \
  --device cuda:0


# validate
python validate/evaluate_train_mod.py \
  --model_directory /data/path/to/training_model/rna004_m6a_mix_ft \
  --directory /data/path/to/mix_dataset \
  --dataset valid \
  --chunks 300000 \
  --valid-chunks 20000 \
  --batchsize 32 \
  --device cuda:0
