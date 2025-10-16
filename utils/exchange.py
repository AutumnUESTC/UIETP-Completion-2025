import tensorflow as tf

from mindspore.train.serialization import load_checkpoint, load_param_into_net
from nets.backbone.ShuffleNetV1 import shufflenet_g3

# 创建MindSpore模型
mindspore_model = shufflenet_g3()

# 加载MindSpore的Checkpoint文件
ckpt_path = 'model_data/shufflenet_v1_g3.ckpt'
param_dict = load_checkpoint(ckpt_path)

# 将参数加载到MindSpore模型中
load_param_into_net(mindspore_model, param_dict)

import torch

# 创建相应的PyTorch模型
pytorch_model = shufflenet_g3()



# 遍历对应的层，将权重从MindSpore模型转移到PyTorch模型
for (mindspore_name, mindspore_param), (pytorch_name, pytorch_param) in zip(mindspore_model.parameters_dict().items(), pytorch_model.named_parameters()):
    # 确保层名称和参数名称匹配
    assert mindspore_name == pytorch_name, "Layer names do not match."

    # 将NumPy数组转换为PyTorch张量
    pytorch_param.data = torch.from_numpy(mindspore_param.asnumpy())


torch.save(pytorch_model.state_dict(), 'model_data/shufflenet_v1_g3.pth')
