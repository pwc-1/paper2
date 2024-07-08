from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor
import torch
 
def pytorch2mindspore(ckpt_name='res18_py.pth'):
 
    par_dict = torch.load(ckpt_name, map_location=torch.device('cpu'))
 
    new_params_list = []
 
    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]
 
        print('========================py_name',name)
 
        if name.endswith('normalize.bias'):
            name = name[:name.rfind('normalize.bias')]
            name = name + 'normalize.beta'
 
        elif name.endswith('normalize.weight'):
            name = name[:name.rfind('normalize.weight')]
            name = name + 'normalize.gamma'
 
        elif name.endswith('.running_mean'):
            name = name[:name.rfind('.running_mean')]
            name = name + '.moving_mean'
 
        elif name.endswith('.running_var'):
            name = name[:name.rfind('.running_var')]
            name = name + '.moving_variance'
 
        print('========================ms_name',name)
 
        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())
        new_params_list.append(param_dict)
 
 
    save_checkpoint(new_params_list,  'res18_ms.ckpt')