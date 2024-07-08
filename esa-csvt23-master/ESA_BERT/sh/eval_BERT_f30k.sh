cd ../
DATASET_NAME='f30k'
DATA_PATH='/opt/data/private/kevin/data/'${DATASET_NAME}
MODEL_NAME='../runs/F30K_BERT/f30k_butd_region_bert1'
CUDA_VISIBLE_DEVICES=0 

python3 eval.py --dataset ${DATASET_NAME}  --data_path ${DATA_PATH} --model_name ${MODEL_NAME}