"""
train_net.py:网络训练/测试启动文件
=================================

本文件是网络训练和测试时的启动文件.

- 训练网络:
- 测试网络:

"""
# 第三方库
import torch.multiprocessing
# 自建库
from lib.config import cfg, args
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network
from lib.evaluators import make_evaluator

def train(cfg, network):
    """
    train 训练网络

    :param cfg: 配置管理器
    :type cfg: CfgNode
    :param network: 待训练网络实例
    :type network: torch.nn.Module
    :return: 经过训练的网络实例
    :rtype: torch.nn.Module
    """
    # 生成训练器trainer,优化器optimizer,调度器scheduler,记录器recorder,评估器evaluator
    if cfg.train.dataset[:4] != 'City':
        torch.multiprocessing.set_sharing_strategy('file_system')
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    evaluator = make_evaluator(cfg)

    # 加载模型参数(继续训练用)
    begin_epoch = load_model(network, optimizer, scheduler, recorder, cfg.model_dir, resume=cfg.resume)

    # 加载训练集和验证集
    train_loader = make_data_loader(cfg, is_train=True, max_iter=cfg.ep_iter)
    val_loader = make_data_loader(cfg, is_train=False)

    # 迭代数据集
    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()

        if (epoch + 1) % cfg.save_ep == 0:
            save_model(network, optimizer, scheduler, recorder, epoch, cfg.model_dir)

        if (epoch + 1) % cfg.eval_ep == 0:
            trainer.val(epoch, val_loader, evaluator, recorder)

    return network


def test(cfg, network):
    """
    test 基于测试集验证网络的性能

    :param cfg: 配置管理器
    :type cfg: CfgNode
    :param network: 待训练网络实例
    :type network: torch.nn.Module
    """
    trainer = make_trainer(cfg, network)  # 生成训练器
    cfg.is_val = True
    val_loader = make_data_loader(cfg, is_train=False)  # 加载验证集
    evaluator = make_evaluator(cfg)   # 加载评估器 
    epoch = load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)  # 加载指定训练次数(cfg.test.epoch)的模型参数 
    trainer.val(epoch, val_loader, evaluator)  # 验证网络性能


def main():
    """
    main 启动函数
    """
    network = make_network(cfg)
    if args.test:
        cfg.is_val = False
        test(cfg, network)
    else:
        cfg.is_val = True
        train(cfg, network)


if __name__ == "__main__":
    main()
