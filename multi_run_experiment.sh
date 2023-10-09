#! /bin/bash
mine_dir=/home/sh2/S3629862/projects/mine


# SASRec LESSR Ours NISER TAGNN NGCF GCSAN GCEGNN GRU4Rec
for model in Ours GCEGNN LESSR NISER GRU4Rec SASRec NGCF GCSAN BERT4Rec OurBERT4Rec TAGNN
do
   python /home/sh2/S3629862/projects/mine/RecBole-GNN/run_recbole_gnn.py -m $model -d diginetica5core --config_files ./recbole_gnn/properties/overall.server.yaml --nproc 1
done