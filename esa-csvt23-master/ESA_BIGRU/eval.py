import os
import argparse
import logging
from lib import evaluation
from lib import evaluation_mindspore

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import mindspore.context as context

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='f30k',
                        help='coco or f30k')
    parser.add_argument('--data_path', default='/opt/data/private/kevin/data/f30k')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--model_name', default='/root/ESA/runs/F30K_BIGRU/f30k_butd_region_bigru_514.7',
                        help='Path to save the model.')
    opt = parser.parse_args()

    context.set_context(device_target="GPU")

    if opt.dataset == 'coco':
        weights_bases = [
            opt.model_name
        ]
    elif opt.dataset == 'f30k':
        weights_bases = [
            opt.model_name
        ]
    else:
        raise ValueError('Invalid dataset argument {}'.format(opt.dataset))

    for base in weights_bases:
        logger.info('Evaluating {}...'.format(base))
        model_path = os.path.join(base, 'model_best.pth')
        if opt.save_results:  # Save the final results for computing ensemble results
            save_path = os.path.join(base, 'results_{}.npy'.format(opt.dataset))
        else:
            save_path = None

        if opt.dataset == 'coco':
            # Evaluate COCO 5-fold 1K
            evaluation_mindspore.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=True)
            # Evaluate COCO 5K
            evaluation_mindspore.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=False, save_path=save_path)
        elif opt.dataset == 'f30k':
            # Evaluate Flickr30K
            # evaluation.evalrank(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)
            evaluation_mindspore.evalrank(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)



if __name__ == '__main__':
    main()
