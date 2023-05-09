"""
lib:PVNet自建库
===============

PVNet包含的子库有:

- config:读取命令行参数,并管理神经网络训练/测试时的配置信息.
- datasets:实现数据集数据的读取,预处理,数据增强,乱序,批量化等功能.
- networks:基于残差网络(ResNet)实现PVNet的搭建,输出目标的实例分割掩码(mask)和投票向量(vote).
- train:基于优化器(optimizer)和调度器(scheduler)训练网络,并使用记录器(recoder)记录网络训练/验证过程中的信息.
- evaluators:实现了多种评价指标用来评价网络的性能.
- visualizers:实现了相关数据可视化的功能.
"""