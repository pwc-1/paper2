import logging
from lib import evaluation_mindspore
import os
import argparse

logging.basicConfig()
logger = logging.getLogger()    
logger.setLevel(logging.INFO)

# save results
os.system("python3 eval.py --dataset f30k --data_path /opt/data/private/kevin/data/f30k --model_name ../runs/F30K_BIGRU/f30k_butd_region_bigru_514.7 --save_results")
os.system("python3 eval.py --dataset f30k --data_path /opt/data/private/kevin/data/f30k --model_name ../runs/F30K_BIGRU/f30k_butd_region_bigru_513.6 --save_results")
# Evaluate model ensemble
paths = ['../runs/F30K_BIGRU/f30k_butd_region_bigru_514.7/results_f30k.npy',
         '../runs/F30K_BIGRU/f30k_butd_region_bigru_513.6/results_f30k.npy']
print('-------------------------------------------------------------------------------------')

evaluation_mindspore.eval_ensemble(results_paths=paths, fold5=False)


# print('---------------------------------coco----------------------------------------------------')
# os.system("python3 eval.py --dataset coco --data_path ../../data/coco --model_name runs/coco_best1 --save_results")
# os.system("python3 eval.py --dataset coco --data_path ../../data/coco --model_name runs/coco_best2  --save_results")
# # Evaluate model ensemble
# paths = ['runs/coco_best1/results_coco.npy',
#          'runs/coco_best2/results_coco.npy']
# print('-------------------------------------------------------------------------------------')
# evaluation.eval_ensemble(results_paths=paths, fold5=True)
# evaluation.eval_ensemble(results_paths=paths, fold5=False)