#! /bin/bash

mine_dir=/home/sh2/S3629862/projects/mine
rm -rv $mine_dir/{RecBole,RecBole-GNN}/{log,log_tensorboard,saved}/*

python $mine_dir/RecBole-GNN/run_recbole_gnn_group.py -m GRU4Rec,SASRec,LESSR,NISER,TAGNN,LightGCN,NGCF,GCSAN,GCEGNN -d ml-1m --config_files ./recbole_gnn/properties/overall.server.yaml --nproc 1