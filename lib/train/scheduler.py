"""
scheduler模块
=============

本模块主要用来生成调度器实例 ``scheduler`` ,以调整神经网络训练过程学习率的大小,辅助优化器 ``optimizer`` 优化网络的参数.
可选的调度器类型有两种:

- ``MultiStepLR`` :变步长调度器,基于milestones中的回合数,变步长的衰减学习率(乘以gamma因数).
- ``WarmupMultiStepLR`` :带预热的变步长调度器,在MultiStepLR的基础上增加了预热的机制.
  在网络开始训练时先采用一个较小的学习率(LearningRate*WarmupFactor),经过指定的回合数(warmup_iters)后再恢复到初始的学习率.

.. note:: ``milestones`` 中的回合数可以重复,表示该回合学习率需要连续进行多次衰减.如:milestones=[10,20,20,20,30],表示在第20个回合的学习
          率LearningRate=LastLearningRate*gamma*gamma*gamma.
"""
# 第三方库
from collections import Counter
# 自建库
from lib.utils.optimizer.lr_scheduler import WarmupMultiStepLR, MultiStepLR


def make_lr_scheduler(cfg, optimizer):
    """
    make_lr_scheduler 生成调度器实例.

    :param cfg: 配置管理器
    :type cfg: CfgNode
    :param optimizer: 优化器
    :type optimizer: torch.optim.Optimizer
    :return: 学习率调度器
    :rtype: torch.optim.lr_scheduler._LRScheduler
    """
    if cfg.train.warmup:
        # 带预热的变步长的学习率调度器
        scheduler = WarmupMultiStepLR(optimizer, cfg.train.milestones, cfg.train.gamma, 1.0 / 3, 5, 'linear')
    else:
        # 不带预热的变步长学习率调度器
        scheduler = MultiStepLR(optimizer, milestones=cfg.train.milestones, gamma=cfg.train.gamma)
    return scheduler