"""
config子模块
============

该模块主要实现两个功能:

- 基于yacs管理项目的配置信息.被管理的配置信息主要包括以下几个方面:

    - 通用配置信息:模型名称,路径信息,迭代次数等
    - 训练配置信息:训练参数,优化器参数(optimizer),调度器参数(scheduler),数据增强参数(augmentation)
    - 测试配置信息:测试/验证参数

- 基于argparse读取命令行参数,并根据这些参数更新配置信息.
"""

# 标准库
import os
import sys
import argparse

#第三方库
from yacs.config import CfgNode as CN

cfg = CN()

# model
cfg.model = 'custom'
"""模型名称(即待测目标)"""

# dir
cfg.model_dir = 'data/model'
"""模型的训练参数保存路径"""
cfg.record_dir = 'data/record'
"""训练/评估信息保存路径"""
cfg.result_dir = 'data/result'
"""结果保存路径"""

# network heads
cfg.heads = ''
"""确定网络输出的特征数量"""

# task
cfg.task = 'pvnet'
"""任务类型(指定训练/测试的网络)"""

# if load the pretrained network
cfg.resume = True
"""是否继续训练模型"""

# evaluation
cfg.skip_eval = False
"""是否启用评估功能"""

# dataset
cfg.cls_type = 'charger'
"""目标名称"""

# epoch
cfg.ep_iter = -1
"""回合内最大迭代次数"""
cfg.save_ep = 1
"""参数保存的回合间隔"""
cfg.eval_ep = 5
"""模型评估的回合间隔"""

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.dataset = 'CustomTrain'
"""训练集名称"""
cfg.train.epoch = 20
"""训练回合"""
cfg.train.num_workers = 8
"""tarin dataloader子线程数"""
cfg.train.batch_size = 16
"""训练批数据大小"""
cfg.train.batch_sampler = 'image_size'
"""训练时的批采样策略"""

# use adam as default
cfg.train.optim = 'adam'
"""采用的优化器类型"""
cfg.train.lr = 1e-3
"""学习率"""
cfg.train.weight_decay = 0.
"""权重衰减率"""

# scheduler
cfg.train.warmup = False
""""""
cfg.train.milestones = [4, 8, 12, 16, 20]
"""里程数(当epoch达到对应的里程时,就对学习率进行衰减)"""
cfg.train.gamma = 0.5
"""学习率衰减比例"""

# augmentation
cfg.train.rotate_min = -30
"""图像旋转时的最小转角"""
cfg.train.rotate_max = 30
"""图像旋转时的最大转角"""

cfg.train.overlap_ratio = 0.8
"""图像放缩时的重叠率"""
cfg.train.resize_ratio_min = 0.8
"""图像的最小放缩倍数"""
cfg.train.resize_ratio_max = 1.2
"""图像的最大放缩倍数"""

# -----------------------------------------------------------------------------
# val and test
# -----------------------------------------------------------------------------

# val
cfg.val = CN()
cfg.is_val = True
"""是否采用验证集"""
cfg.val.dataset = 'CustomVal'
"""验证集名称"""

# test
cfg.test = CN()
cfg.test.dataset = 'CustomTest'
"""测试集名称"""
cfg.test.num_workers = 8
"""test dataloader子线程数"""
cfg.test.batch_size = 1
"""测试时的批数据大小"""
cfg.test.batch_sampler = 'image_size'
"""测试时的批采样策略"""
cfg.test.epoch = -1
"""测试训练了epoch回合的模型"""
cfg.test.icp = False
"""是否采用最近点迭代(评估用)"""
cfg.test.un_pnp = False
"""是否采用概率型PnP计算位姿"""

_heads_factory = {
    'pvnet': CN({'vote_dim': 18, 'seg_dim': 2}),
    'ct_pvnet': CN({'vote_dim': 18, 'seg_dim': 2}),
    'ct': CN({'ct_hm': 30, 'wh': 2})
}
"""网络输出格式工场"""


def parse_cfg(cfg, args):
    """
    parse_cfg 基于cfg生成cfg.heads以及各种路径信息(cfg.model_dir, cfg.record_dir, cfg.result_dir)

    :param cfg: 配置管理器
    :type cfg: CfgNode
    :param args: 命令行参数
    :type args: Namespace
    """
    # 任务类型必须被指定
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # 配置networks heads
    if cfg.task in _heads_factory:
        cfg.heads = _heads_factory[cfg.task]

    # 配置模型、记录、结果路径
    cfg.model_dir = os.path.join(cfg.model_dir, cfg.task, cfg.model)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.model)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.model)


def make_cfg(args):
    """
    make_cfg 基于命令行参数args更新配置管理器cfg

    :param args: 命令行参数
    :type args: Namespace
    :return: 配置管理器
    :rtype: CfgNode
    """
    # 将命令行中尚未匹配的参数提取至opts，并剔除其中错误参数
    opts_idx = [i for i in range(0, len(args.opts), 2) if args.opts[i].split('.')[0] in cfg.keys()]
    opts = sum([[args.opts[i], args.opts[i + 1]] for i in opts_idx], [])
    # 基于opts更新配置cfg
    cfg.merge_from_list(opts)
    # 生成其它配置信息
    parse_cfg(cfg, args)
    return cfg

# 兼容shpinx用
PATH_SPHINX = "/home/administrator/anaconda3/envs/pvnet/bin/sphinx-build"

if sys.argv[0] == PATH_SPHINX:
    args = argparse.Namespace()
else:
    # 利用argparse对命令行参数进行解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', dest='test', default=False)      # "--test" 是否处于训练状态
    parser.add_argument("--type", type=str, default="")                                 # "--type" 指定要训练的数据集(linemod, custom, etc.)或要进行的任务(visualize, evaluate, etc.)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)                 # "opts" 指向其它可选参数
    args = parser.parse_args()
    # 基于命令行参数更新配置信息
    cfg = make_cfg(args)

# ignore warning message
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")