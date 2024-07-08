import mindspore as ms
import mindspore.nn as nn
import numpy as np
import mindspore.ops as ops
from model import *
from mindspore.dataset.vision import Inter
from mindspore.common.initializer import Normal
from mindspore.dataset import Cifar10Dataset,vision, transforms
from mindspore import Tensor
import time
from parase import *
""" 
python ATLGP_cifar_train_up.py  --bs 128  --epoch 60 --eps 0.031 --alpha 0.0196 --beta 0.9 --iters 60 
"""

model = WideResNet(34, 10,10, dropRate=0.)
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
learing_rate_list = ms.Tensor([0.1]*39+[0.01]*19+[0.001])
optimizer = nn.SGD(model.trainable_params(),learning_rate=learing_rate_list)

# 按照batch打包数据
def datapipe(dataset,batch_size):
    image_transforms = [
        vision.Resize(size=32,interpolation=Inter.LINEAR),
        vision.Rescale(1.0 / 255.0, 0.0),
#         vision.Rescale(1 / 0.3081, -1 * 0.1307 / 0.3081),
        vision.HWC2CHW()
    ]
    label_transforms = transforms.TypeCast(ms.int32)
    dataset = dataset.map(image_transforms,'image')
    dataset = dataset.map(label_transforms,'label')
    dataset = dataset.batch(batch_size)
    return dataset

def forward_fn(data,label):
        out = model(data)
        loss = loss_fn(out,label)
        return loss

def get_gradient(inputs, labels):
    grad_fn = ms.value_and_grad(forward_fn,0)
    loss,out_grad = grad_fn(inputs, labels)
    gradient = out_grad.asnumpy()
    gradient = np.sign(gradient)
    return loss,gradient

def atlgp_attack(inputs, labels, eps=0.3, alpha=0.0196, iters=100 ,beta = 0.95, rand = True) :
            
    labels_tensor = ms.Tensor(labels)
    if rand:
        x = inputs + np.random.uniform(-eps, eps, inputs.shape)
        x = np.float32(x)
        x = np.clip(x, 0, 1) # ensure valid pixel range
    else:
        x = np.copy(inputs)
    loss,grad = get_gradient(ms.Tensor(x), labels_tensor)
    loss_before = loss
    for i in range(iters):
        x += alpha * grad
        x = np.clip(x, inputs - eps, inputs + eps) 
        x = np.clip(x,0,1)
        loss_after,grad = get_gradient(ms.Tensor(x), labels_tensor)
        if loss_before/loss_after >= beta :
            break
        else:
            loss_before = loss_after
    return x

def generate(inputs, labels, eps=0.3,alpha=0.12, iters=100 ,beta = 0.9, rand = True):
    # 对数据集进行处理

    x_batch = inputs.asnumpy().astype(np.float32)
    y_batch = labels.asnumpy()
    adv_x = atlgp_attack(x_batch, y_batch, eps=eps,alpha=alpha, iters=iters ,beta=beta , rand=rand )
    return adv_x

def train(model, dataset,eps=0.3,alpha=0.12, iters=100 ,beta = 0.9, rand = True):
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)
    def train_step(data,label):
        loss_before,grads_before = grad_fn(data,label)
        optimizer(grads_before)
        adv_x = generate(data,label,eps=eps,alpha=alpha, iters=iters ,beta=beta , rand=rand )
        adv_x = Tensor.from_numpy(adv_x)
        loss_after,grads_after = grad_fn(adv_x,label)
        optimizer(grads_after)
        return loss_before,loss_after
    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss_before,loss_after = train_step(data, label)
        if batch % 100 == 0:
            loss_before,loss_after, current = loss_before.asnumpy(),loss_after.asnumpy(), batch
            print(f"loss_before: {loss_before:>7f} loss_after: {loss_after:>7f}  [{current:>3d}/{size:>3d}]")
    print()

def test(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main(args):
    BATCH_SIZE=int(args.bs)
    eps=args.eps
    alpha=args.alpha
    beta = args.beta
    iters=args.iters
    rand = args.rand
    if args.trainckpts:
        trainckpts = args.trainckpts    
    else:
        trainckpts = "ckpts/model_cifar_adam.ckpt"
    

    dataset_train = Cifar10Dataset(dataset_dir="./cifar-10-batches-bin", usage="train", shuffle=True)
    dataset_eval = Cifar10Dataset(dataset_dir="./cifar-10-batches-bin", usage="test", shuffle=True)

    dataset_train = datapipe(dataset_train,BATCH_SIZE)
    dataset_eval = datapipe(dataset_eval,BATCH_SIZE)

    # 加载原样本预训练模型
    param_dict = ms.load_checkpoint(trainckpts)
    # param_dict = ms.load_checkpoint("ckpts/model_cifar_adam.ckpt")
    param_not_load = ms.load_param_into_net(model, param_dict)
    print(model)

    epochs = 60
    time_sum = 0
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        time_start = time.time()
        train(model, dataset_train,eps=eps,alpha=alpha, iters=iters ,beta=beta , rand=rand)
        time_end = time.time()
        time_sum +=time_end-time_start
        test(model, dataset_eval, loss_fn)
    print('time used:',time_sum)
    print("Done!")
    
    # Save checkpoint
    ms.save_checkpoint(model, "ckpts/atlgp_cifar.ckpt")
    print("Saved Model to ckpts/atlgp_cifar.ckpt")

if __name__ =='__main__':
    arg = set_arg()
    ms.set_context(device_target='GPU')
    # 数据预处理
    main(arg)


    
       
    
    
    
    
    
    
    
    
    
    
    