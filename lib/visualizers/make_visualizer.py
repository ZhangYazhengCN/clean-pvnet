"""
make_visualizer模块
===================

生成指定的可视化工具.
"""
# 标准库
import os
import imp
# 自建库
from lib.datasets.dataset_catalog import DatasetCatalog


def make_visualizer(cfg, is_train=False):
    """
    make_visualizer 生成可视化工具(visualizer)实例

    :param cfg: 配置管理器
    :type cfg: CfgNode
    :param is_train: 是否可视化训练集数据, 默认值为False
    :type is_train: bool
    :return: 可视化工具实例
    :rtype: Visualizer
    """
    task = cfg.task
    data_source = DatasetCatalog.get(cfg.test.dataset)['id'] 
    module = '.'.join(['lib.visualizers', data_source, task])
    path = os.path.join('lib/visualizers', data_source, task+'.py')
    visualizer = imp.load_source(module, path).Visualizer(is_train)
    return visualizer
