python -m bonito train -f ./bonito_simplecnn \
 --directory /data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10/ \
 --epochs 30 \
 --chunks 100000 \
 --valid-chunks 10000 \
 --config /home/lijy/workspace/bonito/bonito/models/configs/simple_cnn_basecaller.toml


# =========================== Baseline ===================================

python -m bonito train -f /data/biolab-nvme-pcie2/lijy/HG002/bonito_models/bonito_r10_sup@v5.0.0_customdataset \
 --directory /data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10_dorado/ \
 --epochs 30 \
 --chunks 200000 \
 --valid-chunks 20000 \
 --batch 96 \
 --lr 1e-4 \
 --config /home/lijy/workspace/bonito/bonito/models/configs/dna_r10.4.1@v5.0.toml


python -m bonito train -f /data/biolab-nvme-pcie2/lijy/HG002/bonito_models/bonito_r10_sup@v5.0.0_bonitodataset \
 --directory /data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10/ \
 --epochs 30 \
 --chunks 400000 \
 --valid-chunks 40000 \
 --batch 128 \
 --lr 2e-4 \
 --config /home/lijy/workspace/bonito/bonito/models/configs/dna_r10.4.1@v5.0.toml


python -m bonito train -f /data/biolab-nvme-pcie2/lijy/HG002/bonito_models/bonito_r10_sup@v5.2.0_bonitodataset \
 --directory /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10_called/bonito_sup/ \
 --epochs 30 \
 --chunks 400000 \
 --valid-chunks 40000 \
 --batch 32 \
 --lr 5e-4 \
 --config /home/lijy/workspace/bonito-uv/bonito/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0/config.toml


python -m bonito train -f /data/biolab-nvme-pcie2/lijy/HG002/bonito_models/bonito_r10_sup@v5.2.0_doradov5dataset \
 --directory /data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10_dorado_v5/ \
 --epochs 30 \
 --chunks 400000 \
 --valid-chunks 40000 \
 --batch 32 \
 --lr 5e-4 \
 --config /home/lijy/workspace/bonito-uv/bonito/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0/config.toml


python -m bonito train -f /data/biolab-nvme-pcie2/lijy/HG002/bonito_models/bonito_r10_sup@v5.2.0_doradov6dataset \
 --directory /data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10_dorado_v6/ \
 --epochs 30 \
 --chunks 400000 \
 --valid-chunks 40000 \
 --batch 32 \
 --lr 5e-4 \
 --config /home/lijy/workspace/bonito-uv/bonito/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0/config.toml


python -m bonito train -f /data/biolab-nvme-pcie2/lijy/HG002/bonito_models/bonito_r10_sup@v5.2.0_doradov8dataset \
 --directory /data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10_dorado_v8/ \
 --epochs 30 \
 --chunks 400000 \
 --valid-chunks 40000 \
 --batch 32 \
 --lr 5e-4 \
 --config /home/lijy/workspace/bonito-uv/bonito/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0/config.toml


python -m bonito train -f /data/biolab-nvme-pcie2/lijy/HG002/bonito_models/bonito_r10_sup@v5.2.0_doradov8dataset \
 --directory /data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10_dorado_v8/ \
 --epochs 20 \
 --chunks 400000 \
 --valid-chunks 40000 \
 --batch 32 \
 --lr 1e-4 \
 --config /home/lijy/workspace/bonito-uv/bonito/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0/config.toml




# =========================== Multihead ===================================


python -m bonito train_mod -f /data/biolab-nvme-pcie2/lijy/HG002/multihead_models/bonito_r10_sup@v5.2.0_doradov8dataset_0114 \
 --config /home/lijy/workspace/bonito/bonito/models/configs/multihead_transformer.toml \
 --pretrained /home/lijy/workspace/bonito-uv/bonito/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0/ \
 --freeze-conv \
 --directory /data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10_dorado_v8  / \
 --device cuda:0 \
 --epochs 20 \
 --lr 1e-4 \
 --batch 32 \
 --chunks 50000 \
 --valid-chunks 5000 \


# =========================== TEST BASECALL ===================================

bonito basecaller \
 dna_r10.4.1_e8.2_400bps_sup@v5.2.0 \
 --device cuda:0 \
 /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_1 \
 > /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_1_testcalled/bonito_sup/bonito_r10_sup@v5.2.0_std.bam

samtools fastq -T "*" ./bonito_r10_sup@v5.2.0_std.bam | \
minimap2 -ax map-ont -t 16 --secondary=no /data/biolab-nvme-pcie2/lijy/HG002/hg38.fa - | \
samtools sort -o ./bonito_r10_sup@v5.2.0_std_aligned.bam

samtools index ./bonito_r10_sup@v5.2.0_std_aligned.bam


bonito basecaller \
 /home/lijy/workspace/bonito/bonito/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0_doradov8set \
 --device cuda:0 \
 --max-reads 1000 \
 /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_1 \
 > /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_1_testcalled/bonito_sup/doradov8set/bonito_r10_sup@v5.2.0_doradov8dataset.bam

samtools fastq -T "*" ./bonito_r10_sup@v5.2.0_doradov8dataset.bam | \
minimap2 -ax map-ont -t 16 --secondary=no /data/biolab-nvme-pcie2/lijy/HG002/hg38.fa - | \
samtools sort -o ./bonito_r10_sup@v5.2.0_doradov8dataset_aligned.bam

samtools index ./bonito_r10_sup@v5.2.0_doradov8dataset_aligned.bam



bonito basecaller \
 /home/lijy/workspace/bonito/bonito/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0_bonitoset/ \
 --device cuda:0 \
 --max-reads 1000 \
 /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_1 \
 > /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_1_testcalled/bonito_sup/bonito_r10_sup@v5.2.0_bonitoset.bam

samtools fastq -T "*" ./bonito_r10_sup@v5.2.0_bonitoset.bam | \
minimap2 -ax map-ont -t 16 --secondary=no /data/biolab-nvme-pcie2/lijy/HG002/hg38.fa - | \
samtools sort -o ./bonito_r10_sup@v5.2.0_bonitoset_aligned.bam

samtools index ./bonito_r10_sup@v5.2.0_bonitoset_aligned.bam





python -m bonito basecaller \
 /home/lijy/workspace/bonito/bonito/models/multihead_transformer \
 --device cuda:0 \
 --max-reads 1000 \
 --no-use-koi \
 /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_1 \
 > /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_1_testcalled/bonito_sup/bonito_r10_sup@v5.2.0_multihead.bam

samtools fastq -T "*" ./bonito_r10_sup@v5.2.0_bonitoset.bam | \
minimap2 -ax map-ont -t 16 --secondary=no /data/biolab-nvme-pcie2/lijy/HG002/hg38.fa - | \
samtools sort -o ./bonito_r10_sup@v5.2.0_bonitoset_aligned.bam

samtools index ./bonito_r10_sup@v5.2.0_bonitoset_aligned.bam



# =========================== CTC CRF Finetune ===================================

python -m bonito finetune -f /data/biolab-nvme-pcie2/lijy/HG002/bonito_models/bonito_r10_sup@v5.2.0_doradov8dataset_ft \
 --directory /data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10_dorado_v8/ \
 --pretrained /home/lijy/workspace/bonito/bonito/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0/ \
 --epochs 30 \
 --chunks 100000 \
 --valid-chunks 10000 \
 --batch 64 \
 --lr 1e-4 \
 --config /home/lijy/workspace/bonito/bonito/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0_finetune/config.toml

python -m bonito basecaller \
 /home/lijy/workspace/bonito/bonito/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0_v8ft/ \
 --device cuda:0 \
 --max-reads 1500 \
 --no-use-koi \
 /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_1 \
 > /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_1_testcalled/bonito_sup/v8ft/bonito_r10_sup@v5.2.0_v8ft.bam

samtools fastq -T "*" ./bonito_r10_sup@v5.2.0_v8ft.bam | \
minimap2 -ax map-ont -t 16 --secondary=no /data/biolab-nvme-pcie2/lijy/HG002/hg38.fa - | \
samtools sort -o ./bonito_r10_sup@v5.2.0_v8ft_aligned.bam

samtools index ./bonito_r10_sup@v5.2.0_v8ft_aligned.bam




python -m bonito finetune -f /data/biolab-nvme-pcie2/lijy/HG002/bonito_models/bonito_r10_sup@v5.2.0_bonitodataset_ft \
 --directory /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10_called/bonito_sup/ \
 --pretrained /home/lijy/workspace/bonito/bonito/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0/ \
 --epochs 10 \
 --chunks 100000 \
 --valid-chunks 10000 \
 --batch 32 \
 --lr 1e-4 \
 --config /home/lijy/workspace/bonito/bonito/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0_finetune/config.toml


python -m bonito basecaller \
 /home/lijy/workspace/bonito/bonito/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0_boft/ \
 --device cuda:0 \
 --max-reads 1500 \
 --no-use-koi \
 /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_1 \
 > /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_1_testcalled/bonito_sup/bonitoft/bonito_r10_sup@v5.2.0_bonitoft.bam

samtools fastq -T "*" ./bonito_r10_sup@v5.2.0_bonitoft.bam | \
minimap2 -ax map-ont -t 16 --secondary=no /data/biolab-nvme-pcie2/lijy/HG002/hg38.fa - | \
samtools sort -o ./bonito_r10_sup@v5.2.0_bonitoft_aligned.bam

samtools index ./bonito_r10_sup@v5.2.0_bonitoft_aligned.bam




python -m bonito finetune -f /data/biolab-nvme-pcie2/lijy/HG002/bonito_models/bonito_r10_sup@v5.2.0_doradov8dataset_ft_test \
 --directory /data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10_dorado_v8/ \
 --pretrained /home/lijy/workspace/bonito/bonito/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0/ \
 --epochs 30 \
 --chunks 30000 \
 --valid-chunks 3000 \
 --batch 32 \
 --lr 1e-4 \
 --config /home/lijy/workspace/bonito/bonito/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0_finetune/config.toml

python -m bonito basecaller \
 /home/lijy/workspace/bonito/bonito/models/test/ \
 --device cuda:0 \
 --max-reads 100 \
 --no-use-koi \
 /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_1 \
 > /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_1_testcalled/bonito_sup/v8ft_test/bonito_r10_sup@v5.2.0_v8ft_test.bam



# =========================== 2026 03 25 ===================================


bonito train_mod  -f  /data/biolab-nvme-pcie2/lijy/m6A/training_model/rna004_m6a_allmod_ft \
  --directory /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mod/dataset \
  --config /home/lijy/workspace/bonito/bonito/models/multihead_transformer/config.toml \
  --pretrained rna004_130bps_sup@v5.2.0 \
  --freeze-conv \
  --freeze-encoder \
  --epochs 10 \
  --lr 5e-5 \
  --batch 8 \
  --chunks 12346 \
  --valid-chunks 1234 \
  --num-workers 8 \
  --device cuda:0


# 小chunk排查问题

bonito train_mod -f  /data/biolab-nvme-pcie2/lijy/m6A/training_model/rna004_m6a_allmod_ft_test \
  --directory /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mod/dataset \
  --config /home/lijy/workspace/bonito/bonito/models/multihead_transformer/config.toml \
  --pretrained rna004_130bps_sup@v5.2.0 \
  --epochs 10 \
  --lr 1e-4 \
  --batch 8 \
  --chunks 68 \
  --valid-chunks 4 \
  --device cuda:0 \
  > /home/lijy/workspace/bonito/log/rna004_m6a_allmod_ft_test_260325.log 2>&1


# 改到 base head 接 crf

bonito train_mod -f /data/biolab-nvme-pcie2/lijy/m6A/training_model/rna004_m6a_allmod_ft_crf \
  --directory /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mod/dataset \
  --config /home/lijy/workspace/bonito/bonito/models/multihead_transformer/config.toml \
  --pretrained rna004_130bps_sup@v5.2.0 \
  --freeze-conv \
  --freeze-base-head \
  --freeze-encoder \
  --epochs 10 \
  --lr 1e-4 \
  --batch 8 \
  --chunks 68 \
  --valid-chunks 4 \
  --device cuda:0

# validate

python validate/evaluate_train_mod.py \
  --model_directory /data/biolab-nvme-pcie2/lijy/m6A/training_model/rna004_m6a_allmod_ft_crf \
  --directory /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mod/dataset \
  --dataset valid \
  --weights 10 \
  --chunks 68 \
  --valid-chunks 4 \
  --batchsize 4 \
  --device cuda:0 \
  --output-dir /home/lijy/workspace/bonito/validate/res/rna004_m6a_allmod_ft_crf/validate_epoch10


# single file finetuned

bonito train_mod -f /data/biolab-nvme-pcie2/lijy/m6A/training_model/rna004_m6a_allmod_1file_ft_crf \
  --directory /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mod/dataset \
  --config /home/lijy/workspace/bonito/bonito/models/multihead_transformer/config.toml \
  --pretrained rna004_130bps_sup@v5.2.0 \
  --freeze-conv \
  --freeze-base-head \
  --freeze-encoder \
  --epochs 30 \
  --lr 5e-5 \
  --batch 64 \
  --chunks 12346 \
  --valid-chunks 1234 \
  --device cuda:0 \
    > /home/lijy/workspace/bonito/log/rna004_m6a_allmod_ft_1file_crf.log 2>&1

python validate/evaluate_train_mod.py \
  --model_directory /data/biolab-nvme-pcie2/lijy/m6A/training_model/rna004_m6a_allmod_1file_ft_crf \
  --directory /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mod/dataset \
  --dataset valid \
  --weights 30 \
  --chunks 12346 \
  --valid-chunks 1234 \
  --batchsize 16 \
  --device cuda:0 \
  --output-dir /home/lijy/workspace/bonito/validate/res/rna004_m6a_allmod_ft_crf/validate_1file_epoch30


bonito train_mod -f /data/biolab-nvme-pcie2/lijy/m6A/training_model/rna004_m6a_allmod_1file_ft_baselevel \
  --directory /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mod/dataset \
  --config /home/lijy/workspace/bonito/bonito/models/configs/multihead_transformer.toml \
  --pretrained rna004_130bps_sup@v5.2.0 \
  --freeze-conv \
  --freeze-base-head \
  --freeze-encoder \
  --epochs 30 \
  --lr 5e-5 \
  --batch 64 \
  --chunks 12346 \
  --valid-chunks 1234 \
  --device cuda:0 \
    > /home/lijy/workspace/bonito/log/rna004_m6a_allmod_ft_1file_baselevel.log 2>&1


python validate/evaluate_train_mod.py \
  --model_directory /data/biolab-nvme-pcie2/lijy/m6A/training_model/rna004_m6a_allmod_1file_ft_baselevel \
  --directory /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mod/dataset \
  --dataset valid \
  --weights 20 \
  --chunks 12346 \
  --valid-chunks 1234 \
  --batchsize 16 \
  --device cuda:0 \
  --signal-example-limit 8 \
  --output-dir /home/lijy/workspace/bonito/validate/res/rna004_m6a_allmod_ft_crf/validate_1file_baselevel_epoch20



# mod trunk + 4 base mod head

bonito train_mod -f /data/biolab-nvme-pcie2/lijy/m6A/training_model/rna004_m6a_mix_15+15_ft \
  --directory /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mix/dataset/mix_15wmod+15wcan \
  --config /home/lijy/workspace/bonito/bonito/models/configs/multihead_transformer.toml \
  --pretrained rna004_130bps_sup@v5.2.0 \
  --freeze-conv \
  --freeze-encoder \
  --freeze-base-head \
  --epochs 30 \
  --batch 48 \
  --lr 5e-5 \
  --chunks 300000 \
  --valid-chunks 20000 \
  --device cuda:0
