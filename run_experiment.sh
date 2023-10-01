#! /bin/bash

mine_dir=/home/sh2/S3629862/projects/mine
for proj_dir in RecBole RecBole-GNN
do
  for dir_name in log log_tensorboard saved
  do
    dir2del="$mine_dir/$proj_dir/$dir_name"
    echo "Deleting $dir2del"
    rm -rf "${dir2del:?}/*"
  done
done

python /home/sh2/S3629862/projects/mine/RecBole-GNN/run_recbole_gnn_group.py -m GRU4Rec,SASRec,LESSR,NISER,TAGNN,LightGCN,NGCF,GCSAN,GCEGNN -d ml-1m --config_files ./recbole_gnn/properties/overall.server.yaml --nproc 1