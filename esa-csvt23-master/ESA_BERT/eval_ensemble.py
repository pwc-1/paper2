import logging
from lib import evaluation_mindspore
import os

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# save results
os.system("python3 eval.py --dataset f30k --data_path /opt/data/private/kevin/data/f30k --model_name ../runs/F30K_BERT/f30k_butd_region_bert --save_results")
os.system("python3 eval.py --dataset f30k --data_path /opt/data/private/kevin/data/f30k --model_name ../runs/F30K_BERT/f30k_butd_region_bert1 --save_results")
# Evaluate model ensemble
paths = ['../runs/F30K_BERT/f30k_butd_region_bert/results_f30k.npy',
         '../runs/F30K_BERT/f30k_butd_region_bert1/results_f30k.npy']
print('-------------------------------------------------------------------------------------')

evaluation_mindspore.eval_ensemble(results_paths=paths, fold5=False)


# os.system("python3 eval.py --dataset coco --data_path ../../data/coco --model_name runs/coco_butd_region_bert --save_results")
# os.system("python3 eval.py --dataset coco --data_path ../../data/coco --model_name runs/coco_butd_region_bert1 --save_results")
# # Evaluate model ensemble
# paths = ['runs/coco_butd_region_bert/results_coco.npy',
#          'runs/coco_butd_region_bert1/results_coco.npy']
# logger.info('------------------------------------ensemble-------------------------------------------------')
# evaluation.eval_ensemble(results_paths=paths, fold5=True)
# evaluation.eval_ensemble(results_paths=paths, fold5=False)
