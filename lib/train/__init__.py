"""
train:网络训练库
================

主要实现神经网络训练/验证时所需的各项功能:

- ``LossFunction`` :损失函数,平滑L1损失函数(smooth_l1_loss)计算投票向量的损失,交叉熵损失函数(CrossEntropy)计算实例分割掩码的损失.
- ``Optimizer`` :优化器,采用Adam/RAdam/SGD优化算法在训练过程优化神经网络的参数.
- ``Recoder`` :记录器,记录神经网络训练和验证过程中的各项信息.
- ``Scheduler`` :调度器,在网络训练过程中对学习率进行调整.

"""
from .trainers import make_trainer
from .optimizer import make_optimizer
from .scheduler import make_lr_scheduler, set_lr_scheduler
from .recorder import make_recorder

