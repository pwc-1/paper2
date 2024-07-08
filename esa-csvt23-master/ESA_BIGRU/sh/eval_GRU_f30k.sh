cd ../

DATASET_NAME='f30k'
DATA_PATH='/opt/data/private/kevin/data/'${DATASET_NAME}
VOCAB_PATH='/opt/data/private/kevin/data/vocab'
MODEL_NAME='../runs/F30K_BIGRU/f30k_butd_region_bigru_514.7'
CUDA_VISIBLE_DEVICES=0 

python3 eval.py --dataset ${DATASET_NAME}  --data_path ${DATA_PATH} --model_name ${MODEL_NAME}
