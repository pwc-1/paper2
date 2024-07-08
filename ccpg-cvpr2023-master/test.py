import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import load_checkpoint
import mindspore.dataset as ds
from mindspore import Model

from model.ogbase_aug import OGBASE_AUG
from losses.crossentropy_loss import CrossEntropyLoss
from losses.triplet_loss import TripletLoss
from data.ccpg import CCPG_DataSet

CCPG_path = "./data/opengait_data_128"

aug = OGBASE_AUG()
load_checkpoint("./11train.ckpt", net=aug)

dataset_eval = CCPG_DataSet(CCPG_path, mode="Test")
dataset4train = ds.GeneratorDataset(dataset_eval, ["silh_data", "id_label", "cloth_label", "view_label"], shuffle=False)
dataset4train = dataset4train.batch(1)

outputs_data = []
output_id = []
output_cloth = []
output_view = []


print("============== Test Start ==============")

for idx, inputs in enumerate(dataset4train.create_dict_iterator()):
    ipts_data = inputs["silh_data"]
    id_label = inputs["id_label"].asnumpy()
    cloth_label = inputs["cloth_label"].asnumpy()
    view_label = inputs["view_label"].asnumpy()
    output = aug(ipts_data.astype(mindspore.float32)/255.)[0]
    # softmax = nn.Softmax()
    # output = softmax(output)
    output = output.asnumpy()
    outputs_data.append(output)
    output_id.append(id_label)
    output_cloth.append(cloth_label)
    output_view.append(view_label)
    if idx % 1000 == 0 and idx != 0:
        print(idx)
        # break

# save_dict = {"silh_data": np.array(outputs_data), "id_label": np.array(output_id), "cloth_label": np.array(output_cloth), "view_label": np.array(output_view)}
# np.save(save_dict, "output.npy")
# outputs_data = np.squeeze(outputs_data, axis=1)
# outputs_data = np.stack(outputs_data, axis=0)
# print(outputs_data.shape)
save_dict = {"silh_data": np.array(outputs_data), "id_label": np.array(output_id), "cloth_label": np.array(output_cloth), "view_label": np.array(output_view)}
np.save("output.npy", save_dict)

print("============== Test Finished ==============")
