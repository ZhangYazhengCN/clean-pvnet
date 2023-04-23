"""
visualizers.custom.pvnet模块
============================


"""
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfg
import pycocotools.coco as coco
import numpy as np
from lib.utils.pvnet import pvnet_config
import matplotlib.pyplot as plt
from lib.utils import img_utils
import matplotlib.patches as patches
from lib.utils.pvnet import pvnet_pose_utils
import cv2
from pathlib import Path
import sys


mean = pvnet_config.mean
std = pvnet_config.std


class Visualizer:

    def __init__(self,is_tarin = False):
        args = DatasetCatalog.get(cfg.train.dataset) if is_tarin else DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.train_back_paths = Path(args['data_root']+'/background/train_imgs.txt').read_text().split('\n')[:-1]
        self.coco = coco.COCO(self.ann_file)

    def visualize(self, output, batch):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()

        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        K = np.array(anno['K'])

        pose_gt = np.array(anno['pose'])
        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)

        corner_3d = np.array(anno['corner_3d'])
        corner_2d_gt = pvnet_pose_utils.project(corner_3d, K, pose_gt)
        corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)

        _, ax = plt.subplots(1)
        ax.imshow(inp)
        ax.add_patch(patches.Polygon(xy=corner_2d_gt[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='g'))
        ax.add_patch(patches.Polygon(xy=corner_2d_gt[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='g'))
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
        plt.show()

    def visualize_train(self, batch):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0).numpy()
        inp = (inp*255).astype(np.uint8)[...,::-1]
        mask = batch['mask'][0].numpy()
        # vertex = batch['vertex'][0][0].numpy()
        img_id = int(batch['img_id'][0])
        back_img = cv2.resize(cv2.imread(self.train_back_paths[img_id-1],cv2.IMREAD_UNCHANGED),(inp.shape[1],inp.shape[0])).astype(np.uint8)
        kpt_2d = np.array(batch['meta']['kpt_2d'][0],dtype=np.int32)

        display = np.zeros((inp.shape[0]*2,inp.shape[1]*2,inp.shape[2]),dtype=inp.dtype)
        display[:back_img.shape[0],:back_img.shape[1]] = back_img
        display[:mask.shape[0],-mask.shape[1]:] = inp*mask[...,None]
        display[-inp.shape[0]:,:inp.shape[1]] = inp
        oran_inp = inp.copy()
        for maker in kpt_2d[:-1]:
            cv2.drawMarker(oran_inp,tuple(maker),(0,255,0),cv2.MARKER_TILTED_CROSS,markerSize=10,thickness=2,line_type=cv2.LINE_AA)
        cv2.drawMarker(oran_inp,tuple(kpt_2d[-1]),(0,0,255),cv2.MARKER_SQUARE,markerSize=10,thickness=2,line_type=cv2.LINE_AA)
        display[-oran_inp.shape[0]:,-oran_inp.shape[1]:] = oran_inp

        cv2.imshow('display',display)
        k =  cv2.waitKey(0)
        if k == 27:
            sys.exit()
        elif k ==ord('s') or k==ord('S'):
            cv2.imwrite('display',display)

