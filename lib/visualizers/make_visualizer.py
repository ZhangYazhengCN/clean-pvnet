import os
import imp
from lib.datasets.dataset_catalog import DatasetCatalog


def make_visualizer(cfg, is_train=False):
    task = cfg.task
    data_source = DatasetCatalog.get(cfg.test.dataset)['id'] 
    module = '.'.join(['lib.visualizers', data_source, task])
    path = os.path.join('lib/visualizers', data_source, task+'.py')
    visualizer = imp.load_source(module, path).Visualizer(is_train)
    return visualizer
