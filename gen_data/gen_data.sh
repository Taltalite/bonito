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



# basecalling

