from torch.utils.data import Dataset
import albumentations as A
import cv2
import numpy as np
import zmq
from superpoint import sample_descriptors
import pickle
import torch

class DemoDataset(Dataset):
    def __init__(self, cfg, is_training=True):
        super().__init__()
        self.cfg = cfg
        self.is_training = is_training
        self.features = dict()
        self.augs = A.Compose([
            A.GaussianBlur(blur_limit=(1, 3)),
            # A.GaussNoise(),
            # A.ISONoise(always_apply=False, p=0.5),                                                      
            # A.RandomBrightnessContrast(),
        ])
        self.context = None
        vidcap = cv2.VideoCapture('data/ref.mp4')
        self.imgs = []
        while True:  
            success, image = vidcap.read()
            if not success:
                break
            self.imgs.append(image[..., ::-1])
        
            
    def __getitem__(self, idx):
        if self.context is None:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect('tcp://localhost:5555')

        rgb = self.imgs[idx]
        if self.is_training:
            rgb = self.augs(image=rgb)['image']
    
        box_bounds = np.stack([np.array([0, 0]), np.array([rgb.shape[1], rgb.shape[0]])]).astype(int)
        
        center2d = np.mean(box_bounds, 0)
        size = np.array([box_bounds[1, 0] - box_bounds[0, 0], box_bounds[1, 1] - box_bounds[0, 1]])
        
        self.socket.send(pickle.dumps(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)))
        res = pickle.loads(self.socket.recv())
        
        descs = res['raw_descs']
        kps = np.stack(np.meshgrid(np.arange(box_bounds[0, 0], box_bounds[1, 0]), np.arange(box_bounds[0, 1], box_bounds[1, 1])), -1).reshape(-1, 2)
        sub_idx = np.random.randint(kps.shape[0], size=(self.cfg.num_samples,))
        kps = kps[sub_idx]

        offsets = center2d - kps  # N x 2
        offsets /= (np.linalg.norm(offsets, axis=-1, keepdims=True) + 1e-9)
        
        rel_size = np.abs(center2d - kps) / size * 10
        feats = sample_descriptors(torch.from_numpy(kps).float()[None], torch.from_numpy(np.moveaxis(descs, -1, 0)[None]))[0].numpy().T
        
        
        return {
            'rgbs': (np.moveaxis(rgb, [0, 1, 2], [1, 2, 0]) / 255.).astype(np.float32),
            'point_kps': kps.astype(np.float32),
            'point_feats': feats.astype(np.float32),
            'rel_size': rel_size.astype(np.float32),
            'offsets': offsets.astype(np.float32),
        }
    
    def __len__(self):
        return len(self.imgs)