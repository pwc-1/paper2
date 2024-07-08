import mindspore as ms
import mindspore.nn as nn
import numpy as np
import mindspore.ops as ops
from model import *
import mindspore.dataset.vision as transforms
from mindspore.dataset.vision import Inter
from mindspore.dataset import MnistDataset
from parase import *

""" 
python attack_minist.py  --bs 64  --eps 0.3 --alpha 0.12 --iters 40 --atkmod fgsm
"""

ms.set_context(device_target='GPU')

network = Network()
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

# forward propagation
def forward_fn(inputs, targets):
    out = network(inputs)
    loss = net_loss(out, targets)
    return loss

# get gradient
def gradient_func(inputs, labels):
    _grad_all = ops.composite.GradOperation(get_all=True, sens_param=False)
    # 求取梯度
    out_grad = _grad_all(forward_fn)(inputs, labels)[0]
    gradient = out_grad.asnumpy()
    gradient = np.sign(gradient)
    return gradient

# generate adversarial example
def fgsm_attack(inputs, labels, eps):
    # 实现FGSM
    inputs_tensor = ms.Tensor(inputs)
    labels_tensor = ms.Tensor(labels)
    gradient = gradient_func(inputs_tensor, labels_tensor)
    # 产生扰动
    perturbation = eps * gradient
    # 生成受到扰动的图片
    adv_x = inputs + perturbation
    adv_x = np.clip(adv_x, -1 * 0.1307 / 0.3081, (1-1 * 0.1307) / 0.3081)
    return adv_x

def pgd_attack(inputs, labels, eps=0.3, alpha=0.12, iters=40 , rand = True) :
        
    inputs_tensor = ms.Tensor(inputs)
    labels_tensor = ms.Tensor(labels)
    if rand:
        x = inputs + np.random.uniform(-eps, eps, inputs.shape)
        x = np.float32(x)
        x = np.clip(x, 0, 1) # ensure valid pixel range
    else:
        x = np.copy(inputs)

    for i in range(iters):
        grad = gradient_func(ms.Tensor(x), labels_tensor)
        x += alpha * grad
        x = np.clip(x, inputs - eps, inputs + eps) 
        x = np.clip(x, -1 * 0.1307 / 0.3081, (1-1 * 0.1307) / 0.3081) # ensure valid pixxel range
        
    return x

def batch_generate(inputs, labels,batch_size=32, eps=0.3,alpha=0.12, iters=40 , rand = True,mod='fgsm'):
    # 对数据集进行处理
    arr_x = inputs
    arr_y = labels
    len_x = len(inputs)
    batches = int(len_x / batch_size)
    res = []
    if mod == 'fgsm':
        for i in range(batches):
            print("\rbatch:",i,end='')
            x_batch = arr_x[i * batch_size: (i + 1) * batch_size]
            y_batch = arr_y[i * batch_size: (i + 1) * batch_size]
            adv_x = fgsm_attack(x_batch, y_batch, eps=eps)
            res.append(adv_x)
    elif mod == 'pgd':
        for i in range(batches):
            print("\rbatch:",i,end='')
            x_batch = arr_x[i * batch_size: (i + 1) * batch_size]
            y_batch = arr_y[i * batch_size: (i + 1) * batch_size]
            adv_x = pgd_attack(x_batch, y_batch, eps=eps,alpha=alpha,iters=iters,rand=rand)
            res.append(adv_x)
        
    adv_x = np.concatenate(res, axis=0)
    return adv_x

def main(args):
    
    bs=int(args.bs)
    eps=args.eps
    alpha=args.alpha
    iters=args.iters
    rand = args.rand
    atkmod=args.atkmod
    if args.atkckpts:
        atkckpts = args.atkckpts    
    else:
        atkckpts = "ckpts/atlgp_minist.ckpt"
    param_dict = ms.load_checkpoint(atkckpts)
    ms.load_param_into_net(network, param_dict)
    # 模型加载
    images = []
    labels = []
    test_images = []
    test_labels = []
    predict_labels = []


    trans_transform = [
            transforms.Resize(size=32, interpolation=Inter.LINEAR),
            transforms.Rescale(1.0 / 255.0, 0.0),
            transforms.Rescale(1 / 0.3081, -1 * 0.1307 / 0.3081),
            transforms.HWC2CHW(),
    ]
    dataset_eval = MnistDataset(dataset_dir="./MNIST_Data/test", usage="test", shuffle=True)
    dataset_eval = dataset_eval.map(operations=trans_transform, input_columns=["image"])
    dataset_eval = dataset_eval.map(operations=lambda x: x.astype("int32"), input_columns=["label"])
    dataset_eval = dataset_eval.batch(batch_size=32, drop_remainder=True)
    ds_test = dataset_eval.create_dict_iterator(output_numpy=True)

    model = ms.Model(network, loss_fn=net_loss,  metrics={'accuracy'})
    
    
    for data in ds_test:
        images = data['image'].astype(np.float32)
        labels = data['label']
        test_images.append(images)
        test_labels.append(labels)
        pred_labels = np.argmax(model.predict(ms.Tensor(images)).asnumpy(), axis=1)
        predict_labels.append(pred_labels)

    test_images = np.concatenate(test_images)
    predict_labels = np.concatenate(predict_labels)
    true_labels = np.concatenate(test_labels)


    advs = batch_generate(test_images, true_labels, batch_size=bs, eps=eps,alpha=alpha,iters=iters,rand = rand, mmod=atkmod)
#     before
    org_predicts = model.predict(ms.Tensor(test_images)).asnumpy()
    org_predicts = np.argmax(org_predicts, axis=1)
    org_accuracy = np.mean(np.equal(org_predicts, true_labels))
    print('\norg_accuracy:',org_accuracy)

    adv_predicts = model.predict(ms.Tensor(advs)).asnumpy()
    adv_predicts = np.argmax(adv_predicts, axis=1)
    accuracy = np.mean(np.equal(adv_predicts, true_labels))
    print('\nattack_accuracy',accuracy)


    import matplotlib.pyplot as plt

    adv_examples = np.transpose(advs[:10], [0, 2, 3, 1])
    ori_examples = np.transpose(test_images[:10], [0, 2, 3, 1])

    plt.figure(figsize=(10, 3), dpi=120)
    for i in range(10):
        plt.subplot(3, 10, i + 1)
        plt.axis("off")
        plt.imshow(np.squeeze(ori_examples[i]))
        plt.subplot(3, 10, i + 11)
        plt.axis("off")
        plt.imshow(np.squeeze(adv_examples[i]))
    plt.show()

if __name__ == '__main__':
    args = set_arg()
    main(args=args)