"""
run.py:
=======

实用工具启动文件
"""
# 第三方库
import numpy as np
# 自建库
from lib.config import cfg, args

def run_custom():
    """
    run_custom 生成个人数据集
    """
    from tools import make_dataset
    data_root = 'data/custom'
    back_root = 'data/custom/background'
    make_dataset(root=data_root,back_root=back_root)


def run_dataset():
    """
    run_dataset DataLoader迭代训练/验证/测试集
    """
    from lib.datasets import make_data_loader
    import tqdm
    # import torch

    cfg.is_val = True
    cfg.test.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):
        pass
        #  if torch.max(batch['mask']) != 1 or torch.min(batch['mask']) != 0:
        #      print(torch.max(batch['mask']),torch.min(batch['mask']),batch['img_id'])


def run_network():
    """
    run_network 使用PVNet网络处理测试集数据,输出平均处理时间
    """
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    import time

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    cfg.is_val = True
    data_loader = make_data_loader(cfg, is_train=False)
    total_time = 0
    # vertex_size = torch.Size([1, 18, 480, 640])
    # mask_size = torch.Size([1, 480, 640])
    # seg_size = torch.Size([1, 2, 480, 640])
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            output = network(batch['inp'])
            torch.cuda.synchronize()
            total_time += time.time() - start
        # if batch['mask'].size() != mask_size:
        #     print(batch['mask'])
        # if batch['vertex'].size() != vertex_size:
        #     print(batch['vertex'])
        # if output['seg'].size() != seg_size:
        #     print(output['seg'])
        # if batch['vertex'].size() != vertex_size:
        #     print(batch['vertex'])            
    print(total_time / len(data_loader))


def run_evaluate():
    """
    run_evaluate 基于测试集评估PVNet性能,输出评估结果
    """
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network

    torch.manual_seed(0)

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    cfg.is_val = False
    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    for batch in tqdm.tqdm(data_loader):
        inp = batch['inp'].cuda()
        with torch.no_grad():
            output = network(inp)
        evaluator.evaluate(output, batch)
    evaluator.summarize()

def run_camera():
    """
    run_camera 调用笔记本摄像头,实时检测目标并估计6D位姿
    """
    import cv2
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    from lib.datasets.transforms import make_transforms
    from lib.datasets.dataset_catalog import DatasetCatalog
    from lib.utils.pvnet import pvnet_pose_utils

    root = DatasetCatalog().get(cfg.train.dataset)['data_root']
    K = np.loadtxt(root+'/camera.txt')
    fps_3d = np.loadtxt(root+'/fps.txt')
    corner_3d = np.loadtxt(root+'/corner_3d.txt')
    center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
    kpt_3d = np.concatenate([fps_3d,center_3d[None]], axis=0)

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    transforms = make_transforms(cfg,is_train=False)

    ratio = 2.05
    camera = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frameSize = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)*ratio),int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)*ratio))
    video = cv2.VideoWriter('data/video/video.avi',fourcc=fourcc,fps=5,frameSize=frameSize,isColor=True)
    cv2.namedWindow('display')
    while True:
        ret,img =  camera.read()
        img = cv2.flip(img,True)
        input, _, _ = transforms(img.copy())
        output = network(torch.from_numpy(input[None]).cuda())
        mask = (output['mask'][0]).detach().cpu().numpy().astype(np.bool)

        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
        img_pose = img.copy()
        pose = 'no object'
        if not np.all(kpt_2d==0):
            pose = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)
            corner_2d = pvnet_pose_utils.project(corner_3d, K, pose).astype(np.int32)
            # img_in = (corner_2d<[480,640]) * (corner_2d>=[0,0]) 
            # corner_2d = corner_2d[img_in[:,0]*img_in[:,1]]
            img_pose = img.copy()
            cv2.polylines(img_pose,corner_2d[None,[0, 1, 3, 2, 0, 4, 6, 2]],isClosed=True,color=(0,255,0),thickness=2,lineType=cv2.LINE_AA)
            cv2.polylines(img_pose,corner_2d[None,[5, 4, 6, 7, 5, 1, 3, 7]],isClosed=True,color=(0,255,0),thickness=2,lineType=cv2.LINE_AA)

        display = np.ones((frameSize[1],frameSize[0],img.shape[2]),img.dtype)*255
        display[:img.shape[0],:img.shape[1]] = img
        display[:mask.shape[0],-mask.shape[1]:] = img*mask[...,None]
        display[-img_pose.shape[0]:,:img_pose.shape[1]] = img_pose
        cv2.imshow('display',display)
        video.write(display)
        print(pose)
        k =  cv2.waitKey(100)
        if k == 27:
            break
    camera.release()
    video.release()
    cv2.destroyAllWindows()


def run_visualize_train():
    """
    run_visualize_train 可视化训练集数据
    """
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    cfg.train.batch_size = 1
    data_loader = make_data_loader(cfg, is_train=True)
    visualizer = make_visualizer(cfg, is_train=True)
    for batch in tqdm.tqdm(data_loader):
        visualizer.visualize_train(batch)



def run_visualize():
    """可视化测试集数据"""
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer
    import matplotlib.pyplot as plt 

    cfg.is_val = False

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        visualizer.visualize(output, batch)
        

if __name__ == '__main__':
    globals()['run_'+args.type]()

