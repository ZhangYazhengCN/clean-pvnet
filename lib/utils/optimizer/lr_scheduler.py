# 标准库
from collections import Counter
from bisect import bisect_right
# 第三方库
import torch


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """
    WarmupMultiStepLR 带预热的变步长学习率调整.

    :param optimizer: 优化器实例
    :type optimizer: torch.optim.Optimizer
    :param milestones: 里程数(表明需要进行学习衰减的历程,内部元素可重复,)
    :type milestones: list
    :param gamma: 学习率衰减比例, 默认值为0.1
    :type gamma: float
    :param warmup_factor: 预热因数, 默认值为1.0/3
    :type warmup_factor: float
    :param warmup_iters: 预热回合数, 默认值为5
    :type warmup_iters: int
    :param warmup_method: 预热方式:线性 ``linear`` /常量 ``constant`` , 默认值为"linear"
    :type warmup_method: str
    :param last_epoch: 上一回合数, 默认值为-1
    :type last_epoch: int
    """
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=2,
        warmup_method="linear",
        last_epoch=-1,
    ):
        """
        __init__ 初始化函数

        :param optimizer: 优化器实例
        :type optimizer: torch.optim.Optimizer
        :param milestones: 里程数(表明需要进行学习衰减的历程,内部元素可重复,)
        :type milestones: list
        :param gamma: 学习率衰减比例, 默认值为0.1
        :type gamma: float
        :param warmup_factor: 预热因数, 默认值为1.0/3
        :type warmup_factor: float
        :param warmup_iters: 预热回合数, 默认值为5
        :type warmup_iters: int
        :param warmup_method: 预热方式:线性 ``linear`` /常量 ``constant`` , 默认值为"linear"
        :type warmup_method: str
        :param last_epoch: 上一回合数, 默认值为-1
        :type last_epoch: int
        """
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        """里程数"""
        self.gamma = gamma
        """学习率调整比例"""
        self.warmup_factor = warmup_factor
        """预热因数"""
        self.warmup_iters = warmup_iters
        """预热的回合数"""
        self.warmup_method = warmup_method
        """预热方式:常值(constant)/线性(linear)"""
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        get_lr 优化参数在第last_epoch个回合所对应的学习率

        :return: 各参数对应学习率的列表
        :rtype: dict
        """
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class MultiStepLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super(MultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups]
