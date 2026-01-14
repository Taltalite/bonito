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