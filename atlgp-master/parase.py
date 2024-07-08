import argparse
import numpy
# 创建解析步骤
def set_arg():
    parser = argparse.ArgumentParser(description='argparse for train and attack')
    # 添加参数步骤
    parser.add_argument('--bs',type=numpy.int32,default=64)
    parser.add_argument('--epoch', type=numpy.int32, default=60)
    parser.add_argument('--eps', type=numpy.float32, default=0.3)
    parser.add_argument('--alpha', type=numpy.float32, default=0.12)
    parser.add_argument('--beta', type=numpy.float32, default=0.9)
    parser.add_argument('--iters', type=numpy.int32, default=20)
    parser.add_argument('--rand', type=bool, default=True)
    parser.add_argument('--trainckpts',action='store_true')
    
    parser.add_argument('--atkmod', default='fgsm')
    parser.add_argument('--atkckpts',action='store_true')
    
    # 解析参数步骤  
    args = parser.parse_args()
    return args