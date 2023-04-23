"""
visualizers.custom.pvnet模块
============================

主要用来对训练集/测试集数据进行可视化操作:

- 训练集数据 ``Visualizer.visualize_train`` :显示背景图片back_img,原始图片img_raw,合成图片render_img,以及带有关键点标记的合成图片
- 测试集数据 ``Visualizer.visualize`` :显示带有真实边界框(绿色)和预测边界框(蓝色)的图片.

"""
# 标准库
import sys
from pathlib import Path
# 第三方库
import cv2
import numpy as np
import pycocotools.coco as coco
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# 自建库
from lib.config import cfg
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.utils import img_utils
from lib.utils.pvnet import pvnet_config
from lib.utils.pvnet import pvnet_pose_utils

mean = pvnet_config.mean
"""图片归一化的均值"""
std = pvnet_config.std
"""图片归一化的方差"""

class Visualizer:
    """
    Visualizer 数据集可视化工具,用于对训练集/测试集数据进行可视化操作. 

    :param is_tarin: 是否可视化训练集数据, 默认值为False
    :type is_tarin: bool
    """
    def __init__(self,is_tarin = False):
        """
        __init__ 初始化函数

        :param is_tarin: 是否可视化训练集数据, 默认值为False
        :type is_tarin: bool
        """
        args = DatasetCatalog.get(cfg.train.dataset) if is_tarin else DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        """数据集注释文件路径"""
        self.train_back_paths = Path(args['data_root']+'/background/train_imgs.txt').read_text().split('\n')[:-1]
        """训练集背景图片列表"""
        self.coco = coco.COCO(self.ann_file)
        """数据集coco注释"""

    def visualize(self, output, batch):
        """
        visualize 通用可视化工具.基于神经网络的输入(batch)输出(output)数据,计算目标的真实角点和预测角点.
        然后在图片上绘制真实边界框和预测边界框,并显示图片.

        :param output: 神经网络输出数据
        :type output: dict
        :param batch: 神经网络输入数据
        :type batch: dict
        """
        # 读取反归一化的图片(inp)和预测的2D关键点(kpt_2d)
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()

        # 读取3D关键点kpt_3d及相机内参K
        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        K = np.array(anno['K'])

        # 读取目标真实位姿pose_gt,PnP算法计算目标预测位姿pose_pred
        pose_gt = np.array(anno['pose'])
        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)

        # 计算目标真实角点corner_2d_gt和预测角点corner_2d_pred
        corner_3d = np.array(anno['corner_3d'])
        corner_2d_gt = pvnet_pose_utils.project(corner_3d, K, pose_gt)
        corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)

        # 显示图片,并基于真实角点和预测角点绘制真实边界框和预测边界框
        _, ax = plt.subplots(1)
        ax.imshow(inp)
        ax.add_patch(patches.Polygon(xy=corner_2d_gt[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='g'))
        ax.add_patch(patches.Polygon(xy=corner_2d_gt[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='g'))
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
        plt.show()

    def visualize_train(self, batch):
        """
        visualize_train 训练集可视化工具.显示训练集的背景图片back_img,原始图片mask*inp,合成图片inp,以及标记有2D关键点的合成图片.
        并根据键盘的输入('s','S')对图片进行保存.

        :param batch: 批量数据
        :type batch: dict
        """
        # 读取合成图片inp,掩码mask,背景图片back_img,2D关键点kpt_2d
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0).numpy()
        inp = (inp*255).astype(np.uint8)[...,::-1]
        mask = batch['mask'][0].numpy()
        # vertex = batch['vertex'][0][0].numpy()
        img_id = int(batch['img_id'][0])
        back_img = cv2.resize(cv2.imread(self.train_back_paths[img_id-1],cv2.IMREAD_UNCHANGED),(inp.shape[1],inp.shape[0])).astype(np.uint8)
        kpt_2d = np.array(batch['meta']['kpt_2d'][0],dtype=np.int32)

        # 绘制inp,mask*inp,back_img,kpt_2d在display上面
        display = np.zeros((inp.shape[0]*2,inp.shape[1]*2,inp.shape[2]),dtype=inp.dtype)
        display[:back_img.shape[0],:back_img.shape[1]] = back_img
        display[:mask.shape[0],-mask.shape[1]:] = inp*mask[...,None]
        display[-inp.shape[0]:,:inp.shape[1]] = inp
        oran_inp = inp.copy()
        for maker in kpt_2d[:-1]:
            cv2.drawMarker(oran_inp,tuple(maker),(0,255,0),cv2.MARKER_TILTED_CROSS,markerSize=10,thickness=2,line_type=cv2.LINE_AA)
        cv2.drawMarker(oran_inp,tuple(kpt_2d[-1]),(0,0,255),cv2.MARKER_SQUARE,markerSize=10,thickness=2,line_type=cv2.LINE_AA)
        display[-oran_inp.shape[0]:,-oran_inp.shape[1]:] = oran_inp

        # 显示display,并根据键盘输入选择是否保存图片
        cv2.imshow('display',display)
        k =  cv2.waitKey(0)
        if k == 27:
            sys.exit()
        elif k ==ord('s') or k==ord('S'):
            cv2.imwrite('display',display)

