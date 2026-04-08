  最短命令指南

  1. 生成 mod 数据集

  mkdir -p /data/.../bonito_rna004/mod_dataset

  bonito basecaller /home/lijy/workspace/bonito/bonito/models/rna004_130bps_sup@v5.2.0 \
    /data/.../mod_pod5 \
    --reference /data/.../hg38.fa \
    --save-ctc \
    --rna \
    --device cuda:0 \
    --batchsize 128 \
    --chunksize 12000 \
    --overlap 600 \
    --min-accuracy-save-ctc 0.99 \
    > /data/.../bonito_rna004/mod_dataset/calls.bam

  python gen_data/make_mod_targets_m6a.py \
    --dataset-dir /data/.../bonito_rna004/mod_dataset \
    --mode full-mod \
    --non-a-policy ignore

  2. 生成 canonical 数据集

  mkdir -p /data/.../bonito_rna004/canonical_dataset

  bonito basecaller /home/lijy/workspace/bonito/bonito/models/rna004_130bps_sup@v5.2.0 \
    /data/.../canonical_pod5 \
    --reference /data/.../hg38.fa \
    --save-ctc \
    --rna \
    --device cuda:0 \
    --batchsize 128 \
    --chunksize 12000 \
    --overlap 600 \
    --min-accuracy-save-ctc 0.99 \
    > /data/.../bonito_rna004/canonical_dataset/calls.bam

  python gen_data/make_mod_targets_m6a.py \
    --dataset-dir /data/.../bonito_rna004/canonical_dataset \
    --mode canonical \
    --non-a-policy ignore

  3. 合并成 train_mod 用数据集

  python gen_data/merge_mod_datasets.py \
    --full-mod-dir /data/.../bonito_rna004/mod_dataset \
    --canonical-dir /data/.../bonito_rna004/canonical_dataset \
    --output-dir /data/.../bonito_rna004/mix_dataset

  4. 训练

  bonito train_mod -f /data/.../training_model/rna004_m6a_mix_ft \
    --directory /data/.../bonito_rna004/mix_dataset \
    --config /home/lijy/workspace/bonito/bonito/models/configs/multihead_transformer.toml \
    --pretrained /home/lijy/workspace/bonito/bonito/models/rna004_130bps_sup@v5.2.0 \
    --epochs 30 \
    --batch 48 \
    --lr 5e-5 \
    --chunks 300000 \
    --valid-chunks 20000 \
    --device cuda:0

  5. 评价

  python validate/evaluate_train_mod.py \
    --model_directory /data/.../training_model/rna004_m6a_mix_ft \
    --directory /data/.../bonito_rna004/mix_dataset \
    --dataset valid \
    --chunks 300000 \
    --valid-chunks 20000 \
    --batchsize 32 \
    --device cuda:0

  如果你要，我下一步可以直接给你一个针对 gen_data/create_dataset_mpv8.py 的最小修补方案，优先改这两点：

  - 先裁 signal[ts:ns] 再切 chunk
  - 正确解码/展开 mv，不要假设只有 0/1