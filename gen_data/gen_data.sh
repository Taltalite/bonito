python create_dataset_mpv8.py \
    --bam-file /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mod/PAW43156_92158b33_73a20312_4.sorted.bam \
    --pod5-dir /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mod/src_data/ \
    --reference-fasta /data/biolab-nvme-pcie2/lijy/HG002/hg38.fa \
    --output-dir /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mod/dataset/ \
    --sample-type rna \
    --max-chunks 100000 \
    --workers 8 \
    --chunk-len 12000 \
    --overlap 600 \
    --norm-strategy pa \
    --pa-mean 79.17339964465278 \
    --pa-std 16.929280371741893

python make_mod_targets_m6a.py \
    --dataset-dir /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mod/dataset/



# m6A mix/mod

python create_dataset_mpv8.py \
    --bam-file /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mix/mod_bam/PAW43156_92158b33_73a20312_0+10.sorted.bam \
    --pod5-dir /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mix/mod_pod5/ \
    --reference-fasta /data/biolab-nvme-pcie2/lijy/HG002/hg38.fa \
    --output-dir /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mix/dataset/mod_PAW43156_92158b33_73a20312_0+10/ \
    --sample-type rna \
    --max-chunks 300000 \
    --workers 12 \
    --chunk-len 12000 \
    --overlap 600 \
    --norm-strategy pa \
    --pa-mean 79.17339964465278 \
    --pa-std 16.929280371741893

python make_mod_targets_m6a.py \
  --dataset-dir /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mix/dataset/mod_PAW43156_92158b33_73a20312_0+10/ \
  --mode full-mod \
  --non-a-policy ignore


# m6A mix/canonical

python create_dataset_mpv8.py \
    --bam-file /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mix/canonical_bam/PAW51322_0f2f3583_34a338cb_0+10.sorted.bam \
    --pod5-dir /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mix/canonical_pod5/ \
    --reference-fasta /data/biolab-nvme-pcie2/lijy/HG002/hg38.fa \
    --output-dir /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mix/dataset/canonical_PAW51322_0f2f3583_34a338cb_0+10/ \
    --sample-type rna \
    --max-chunks 300000 \
    --workers 15 \
    --chunk-len 12000 \
    --overlap 600 \
    --norm-strategy pa \
    --pa-mean 79.17339964465278 \
    --pa-std 16.929280371741893

python make_mod_targets_m6a.py \
  --dataset-dir /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mix/dataset/canonical_PAW51322_0f2f3583_34a338cb_0+10/ \
  --mode canonical \
  --non-a-policy ignore


# merge to mix dataset

python merge_mod_datasets.py \
  --full-mod-dir /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mix/dataset/mod_PAW43156_92158b33_73a20312_0+10/ \
  --canonical-dir /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mix/dataset/canonical_PAW51322_0f2f3583_34a338cb_0+10/ \
  --output-dir /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mix/dataset/mix_15wmod+15wcan