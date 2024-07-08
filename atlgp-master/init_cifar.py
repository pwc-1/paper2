import mindspore 
from mindspore import nn
from mindspore.dataset import vision, transforms
from mindspore.dataset import Cifar10Dataset
from model import *
import math as math
# 按照batch打包数据
def datapipe(dataset,batch_size):
    image_transforms = [
        vision.Resize(size=32,interpolation=vision.Inter.LINEAR),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Rescale(1 / 0.3081, -1 * 0.1307 / 0.3081),
        vision.HWC2CHW()
    ]
    label_transforms = transforms.TypeCast(mindspore.int32)
    dataset = dataset.map(image_transforms,'image')
    dataset = dataset.map(label_transforms,'label')
    dataset = dataset.batch(batch_size)
    return dataset


if __name__ =='__main__':
        
    mindspore.set_context(device_target='GPU')
    # 数据预处理
    dataset_train = Cifar10Dataset(dataset_dir="./cifar-10-batches-bin", usage="train", shuffle=True)
    dataset_eval = Cifar10Dataset(dataset_dir="./cifar-10-batches-bin", usage="test", shuffle=True)

    dataset_train = datapipe(dataset_train,64)
    dataset_eval = datapipe(dataset_eval,64)
    
    model = WideResNet(34, 10,10, dropRate=0.3)
#     print(model)
    
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)
    
    def forward_fn(data,label):
        out = model(data)
        loss = loss_fn(out,label)
        return loss,out
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    
    def train_step(data,label):
        (loss,_),grads = grad_fn(data,label)
        optimizer(grads)
        return loss
    def train(model, dataset):
        size = dataset.get_dataset_size()
        model.set_train()
        for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
            loss = train_step(data, label)

#             if batch % 100 == 0:
#                 loss, current = loss.asnumpy(), batch
#                 print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")
    
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
    
    epochs = 60
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(model, dataset_train)
        test(model, dataset_eval, loss_fn)
    print("Done!")
    
    # Save checkpoint
    mindspore.save_checkpoint(model, "ckpts/model_cifar_adam.ckpt")
    print("Saved Model to ckpts/model_cifar_adam.ckpt")


