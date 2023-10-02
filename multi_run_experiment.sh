#! /bin/bash
mine_dir=/home/sh2/S3629862/projects/mine


# SASRec
for model in LESSR NISER TAGNN LightGCN NGCF GCSAN GCEGNN GRU4Rec
do
   python /home/sh2/S3629862/projects/mine/RecBole-GNN/run_recbole_gnn.py -m $model -d yelp --config_files ./recbole_gnn/properties/overall.server.yaml --nproc -1
done