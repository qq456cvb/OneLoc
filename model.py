import cv2
import torch.nn as nn
import torch
import cupy as cp
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from voting import vote_kernel
from scipy.spatial import KDTree


class ResLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, bn=False, dropout=False) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(dim_in, dim_out)
        if bn:
            self.bn1 = torch.nn.BatchNorm1d(dim_out)
        else:
            self.bn1 = lambda x: x
        self.fc2 = torch.nn.Linear(dim_out, dim_out)
        if bn:
            self.bn2 = torch.nn.BatchNorm1d(dim_out)
        else:
            self.bn2 = lambda x: x
        if dim_in != dim_out:
            self.fc0 = torch.nn.Linear(dim_in, dim_out)
        else:
            self.fc0 = None
        self.dropout = nn.Dropout(0.2) if dropout else nn.Identity()
    
    def forward(self, x):
        x_res = x if self.fc0 is None else self.fc0(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return self.dropout(x + x_res)


class Model(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        fcs = [256] + [128] * 20 + [3 + 2 + 2 + (2 if cfg.vote_corners else 0)]
        self.out_layer = nn.Sequential(
            *[ResLayer(fcs[i], fcs[i + 1], False) for i in range(len(fcs) - 1)]
        )
        extractor_model = SuperPoint({
            'descriptor_dim': 256,
            'nms_radius': 3,
            'max_keypoints': 4096,
            'keypoints_threshold': 0.6
        })
        extractor_model.cuda()
        extractor_model.eval()
        load_network(extractor_model, 'data/models/extractors/SuperPoint/superpoint_v1.pth', force=True)
        self.extractor = extractor_model
        self.cache = dict()
        
    def forward(self, inputs):
        feats = inputs['point_feats'].reshape(-1, 256) # B x N x 256
        # target = inputs['name_idx'][:, None].expand(-1, feats.shape[1]).reshape(-1)
        gt_colors = inputs['point_colors'].reshape(-1, 3)  # [0, 1]
        gt_offsets = inputs['offsets'].reshape(-1, 2)
        gt_rel_size = inputs['rel_size'].reshape(-1, 2)
        
        out = self.out_layer(feats)
        pred_colors = out[..., :3]
        if self.cfg.vote_corners:
            pred_offsets = torch.stack([out[..., 3:5], out[..., 7:9]], -2)
        else:
            pred_offsets = out[..., 3:5]
        pred_offsets = F.normalize(pred_offsets, p=2, dim=-1).reshape(-1, 2)
        pred_rel_size = out[..., 5:7]
        
        loss = {
            # 'obj': F.mse_loss(pred_colors[..., :3], gt_colors[..., :3]),
            'size': F.mse_loss(pred_rel_size, gt_rel_size),
            'unit_offset': -(torch.sum(pred_offsets * gt_offsets, -1) ** 2).mean()
        }
        return loss
    
    @torch.no_grad()
    def predict(self, rgb):
        
        img_size = min(rgb.shape[1], rgb.shape[0])
        kp_feats = self.extractor(torch.from_numpy((cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) / 255.).astype(np.float32)).cuda()[None, None])
        raw_descs = np.moveaxis(kp_feats['raw_descs'][0].cpu().numpy(), 0, -1)

        
        if tuple([*rgb.shape[:2]]) not in self.cache:
            stratum_size = img_size / self.cfg.kp_eq_factor
            x = np.arange(stratum_size / 2, rgb.shape[1], stratum_size)
            y = np.arange(stratum_size / 2, rgb.shape[0], stratum_size)
            kps = np.stack(np.meshgrid(x, y), -1).reshape(-1, 2)
            kps += np.random.rand(*kps.shape) * stratum_size - stratum_size / 2
            kps = kps.astype(int)
            kps[:, 0] = np.minimum(kps[:, 0], rgb.shape[1] - 1)
            kps[:, 1] = np.minimum(kps[:, 1], rgb.shape[0] - 1)
            kdtree = KDTree(kps)
            pair_idxs = np.array(list(kdtree.query_pairs(img_size // self.cfg.knn_rad_factor)))
            self.cache[tuple([*rgb.shape[:2]])] = {
                'kps': kps,
                'pair_idxs': pair_idxs
            }
        else:
            kps, pair_idxs = self.cache[tuple([*rgb.shape[:2]])]['kps'], self.cache[tuple([*rgb.shape[:2]])]['pair_idxs']
            
        descs = sample_descriptors(torch.from_numpy(kps).float()[None], torch.from_numpy(np.moveaxis(raw_descs, -1, 0)[None]))[0].numpy().T
        out = self.out_layer(torch.from_numpy(descs).cuda())
        pred_offsets = F.normalize(out[..., 3:5], p=2, dim=-1).cpu().numpy()
        
        colors = rgb[kps[:, 1], kps[:, 0]] / 255.
        colors = colors[..., :3]
        weights = np.ones((kps.shape[0],), dtype=np.float32)
        
        pred_sizes = out[..., 5:7]

        block_size = (pair_idxs.shape[0] + 512 - 1) // 512
        grid_intvl = img_size // self.cfg.ds_factor
        grid_obj = cp.asarray(np.zeros((rgb.shape[1] // grid_intvl, rgb.shape[0] // grid_intvl), dtype=np.float32))
        grid_cnt = cp.zeros_like(grid_obj)
        grid_size = cp.asarray(np.zeros((rgb.shape[1] // grid_intvl, rgb.shape[0] // grid_intvl, 2), dtype=np.float32))
        vote_kernel(
            (block_size, 1, 1),
            (512, 1, 1),
            (
                cp.ascontiguousarray(cp.array(kps)).astype(cp.float32), 
                cp.ascontiguousarray(cp.array(pred_offsets)).astype(cp.float32), 
                cp.ascontiguousarray(cp.array(pred_sizes)).astype(cp.float32), 
                cp.ascontiguousarray(cp.array(pair_idxs)).astype(cp.int32),
                cp.int32(pair_idxs.shape[0]), 
                grid_obj,
                grid_size,
                grid_cnt,
                grid_obj.shape[0],
                grid_obj.shape[1],
                cp.int32(rgb.shape[1]),
                cp.int32(rgb.shape[0]),
                cp.float32(grid_intvl),
                cp.ascontiguousarray(cp.array(weights)).astype(cp.float32),
            )
        )
        res = grid_obj.get().T
        grid_size = grid_size.get().transpose((1, 0, 2))
        grid_cnt = grid_cnt.get().T
        grid_size = grid_size / (grid_cnt[..., None] + 1e-7)
        
        loc = np.array(np.unravel_index(np.argmax(res), res.shape))[::-1]
        pred_size = grid_size[loc[1], loc[0]] * rgb.shape[0] * 5   # hardcoded, size is 10 times bigger
        pred_size[0] = min(pred_size[0], rgb.shape[1] // 2)
        pred_size[1] = min(pred_size[1], rgb.shape[0] // 2)
        
        loc = loc * grid_intvl + grid_intvl // 2
        
        cv2.circle(rgb, (int(loc[0]), int(loc[1])), 5, (255, 0, 0), -1)
        cv2.rectangle(rgb, (int(loc[0] - pred_size[0]), int(loc[1] - pred_size[1])), 
                      (int(loc[0] + pred_size[0]), int(loc[1] + pred_size[1])), (0, 0, 255), 5)
        
