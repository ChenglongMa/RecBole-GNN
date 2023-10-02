#! /bin/bash

mine_dir=/home/sh2/S3629862/projects/mine

source ./cleanup.sh

# GRU4Rec,SASRec,LESSR,NISER,TAGNN,LightGCN,
python $mine_dir/RecBole-GNN/run_recbole_gnn_group.py -m SASRec,LESSR,NISER,TAGNN,NGCF,GCSAN,GCEGNN,GRU4Rec -d yelp --config_files ./recbole_gnn/properties/overall.server.yaml --nproc -1
# python $mine_dir/RecBole-GNN/run_recbole_gnn.py -m NGCF -d ml-1m --config_files ./recbole_gnn/properties/overall.server.yaml --nproc 1
# python $mine_dir/RecBole-GNN/run_recbole_gnn.py -m GCSAN -d ml-1m --config_files ./recbole_gnn/properties/overall.server.yaml --nproc 1
# python $mine_dir/RecBole-GNN/run_recbole_gnn.py -m GCEGNN -d ml-1m --config_files ./recbole_gnn/properties/overall.server.yaml --nproc 1