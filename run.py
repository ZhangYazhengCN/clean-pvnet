"""
PVNet主启动文件

"""
from lib.config import cfg, args
import numpy as np
import os


def run_custom():
    from tools import make_dataset
    data_root = 'data/custom'
    back_root = 'data/custom/background'
    make_dataset(root=data_root,back_root=back_root)


def run_dataset():
    from lib.datasets import make_data_loader
    import tqdm
    # import torch

    cfg.is_val = True
    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):
        pass
        #  if torch.max(batch['mask']) != 1 or torch.min(batch['mask']) != 0:
        #      print(torch.max(batch['mask']),torch.min(batch['mask']),batch['img_id'])


def run_network():
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
    import cv2
    import torch
    from pycocotools.coco import COCO
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
    video = cv2.VideoWriter('data/video.avi',fourcc=fourcc,fps=5,frameSize=frameSize,isColor=True)
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
            cv2.polylines(img_pose,corner_2d[None,[0, 1, 3, 2, 0, 4, 6, 2],None,:],isClosed=True,color=(0,255,0),thickness=2)
            cv2.polylines(img_pose,corner_2d[None,[5, 4, 6, 7, 5, 1, 3, 7],None,:],isClosed=True,color=(0,255,0),thickness=2)

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
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        visualizer.visualize(output, batch)


def run_net_utils():
    from lib.utils import net_utils
    import torch
    import os

    model_path = 'data/model/rcnn_snake/rcnn/139.pth'
    pretrained_model = torch.load(model_path)
    net = pretrained_model['net']
    net = net_utils.remove_net_prefix(net, 'dla.')
    net = net_utils.remove_net_prefix(net, 'cp.')
    pretrained_model['net'] = net
    model_path = 'data/model/rcnn_snake/rcnn/139.pth'
    os.system('mkdir -p {}'.format(os.path.dirname(model_path)))
    torch.save(pretrained_model, model_path)


def run_render():
    from lib.utils.renderer import opengl_utils
    from lib.utils.vsd import inout
    from lib.utils.linemod import linemod_config
    import matplotlib.pyplot as plt

    obj_path = 'data/linemod/cat/cat.ply'
    model = inout.load_ply(obj_path)
    model['pts'] = model['pts'] * 1000.
    im_size = (640, 300)
    opengl = opengl_utils.NormalRender(model, im_size)

    K = linemod_config.linemod_K
    pose = np.load('data/linemod/cat/pose/pose0.npy')
    depth = opengl.render(im_size, 100, 10000, K, pose[:, :3], pose[:, 3:] * 1000)

    plt.imshow(depth)
    plt.show()


def run_detector_pvnet():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        visualizer.visualize(output, batch)

if __name__ == '__main__':
    globals()['run_'+args.type]()

