#!/bin/bash
DEPLOY="deploy_TRT"
mkdir -p $DEPLOY
bash deploy_base_all.sh $DEPLOY TRT_YOLOv4
bash deploy_base_all.sh $DEPLOY TRT_YOLOv3
bash deploy_base_all.sh $DEPLOY TRT_CPN
bash deploy_base_all.sh $DEPLOY TRT_ResNet50
bash deploy_base_all.sh $DEPLOY TRT_ResNet50_Feature
cp -r utilities/gpu_memory_checker.sh $DEPLOY/linux/utilities/

7z a ../data/deploy_TRT_all.7z $DEPLOY
rm -rf $DEPLOY
