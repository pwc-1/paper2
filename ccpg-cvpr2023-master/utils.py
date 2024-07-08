from __future__ import absolute_import
import os
import sys
import errno
import shutil
import json
import os.path as osp

import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from PIL import Image

def tensor_flip(x, dim):
    xsize = x.shape
    dim = len(x.shape) + dim if dim < 0 else dim
    x = x.reshape(-1, *xsize[dim:])
    x = x.reshape(x.shape[0], x.shape[1], -1)[:, P.TensorGetSlice()(x.shape[1]-1, -1, -1).astype('int32'), :]
    return x.reshape(xsize)

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

from mindspore.train.serialization import save_checkpoint
import shutil
import os.path as osp

def save_checkpoint_mindspore(state, is_best, fpath='checkpoint.ckpt'):
    mkdir_if_missing(osp.dirname(fpath))
    save_checkpoint(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.ckpt'))


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


iimport mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import context, Model
from mindspore.common.parameter import Parameter

class GuidedBackprop():
    """
    Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        self.model.set_grad()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(grad):
            self.gradients = grad
        # Modify the first layer to get its gradient
        first_layer = list(self.model._cells_and_names())[0][1]
        if isinstance(first_layer, nn.Conv2d):
            first_layer.grad = hook_function

    def update_relus(self):
        """
        Updates relu activation functions so that
        1- stores output in forward pass
        2- imputes zero for gradient values that are less than zero
        """
        # Modify the ReLU layers
        for name, module in self.model._cells_and_names():
            if isinstance(module, nn.ReLU):
                module.forward = self.relu_forward_hook_function(module.forward)
                module.backward = self.relu_backward_hook_function(module.backward)

    def relu_backward_hook_function(self, original_backward):
        """
        If there is a negative gradient, change it to zero
        """
        def backward(*args):
            grad_input = original_backward(*args)
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * P.Clamp()(grad_input[0], 0.0, float('inf'))
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)
        return backward

    def relu_forward_hook_function(self, original_forward):
        """
        Store results of forward pass
        """
        def forward(*args):
            output = original_forward(*args)
            self.forward_relu_outputs.append(output)
            return output
        return forward

    def generate_gradients(self, input_image, target_class, cnn_layer, filter_pos):
        self.model.clear_gradients()
        # Forward pass
        x = input_image
        for index, (name, layer) in enumerate(self.model._cells_and_names()):
            x = layer(x)
            if name == cnn_layer:
                break
        activation = P.ReduceSum(keep_dims=False)(P.ReduceSum(keep_dims=False)(x, 3), 2)
        _, idx = P.ArgMaxWithValue()(activation, 1)
        conv_output = P.ReduceSum()(P.Abs(x[0, idx]))
        # Backward pass
        conv_output.backward()
        # Convert MindSpore tensor to numpy array
        gradients_as_arr = self.gradients.asnumpy()
        return gradients_as_arr[0]

def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    path_to_file = os.path.join('../results', file_name + '.jpg')
    save_image(gradient, path_to_file)

def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

from PIL import Image
def save_image(im, path):
    """
        Saves a numpy matrix of shape D(1 or 3) x W x H as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image

    TODO: Streamline image saving, it is ugly.
    """
    if isinstance(im, np.ndarray):
        if len(im.shape) == 2:
            im = np.expand_dims(im, axis=0)
            print('A')
            print(im.shape)
        if im.shape[0] == 1:
            # Converting an image with depth = 1 to depth = 3, repeating the same values
            # For some reason PIL complains when I want to save channel image as jpg without
            # additional format in the .save()
            print('B')
            im = np.repeat(im, 3, axis=0)
            print(im.shape)
            # Convert to values to range 1-255 and W,H, D
        # A bandaid fix to an issue with gradcam
        if im.shape[0] == 3 and np.max(im) == 1:
            im = im.transpose(1, 2, 0) * 255
        elif im.shape[0] == 3 and np.max(im) > 1:
            im = im.transpose(1, 2, 0)
        im = Image.fromarray(im.astype(np.uint8))
    im.save(path)







