from pathlib import Path
import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset
from dataset import OnePoseRawDataset
from torch.optim import Adam, SGD, AdamW
import hydra
import cv2
from model import ModelRaw
import os
import time
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from glob import glob


def load_intrinsic(name):
    lines = open(name).read().splitlines()
    intrinsic = np.eye(3)
    intrinsic[0, 0] = float(lines[0].split()[-1])
    intrinsic[1, 1] = float(lines[1].split()[-1])
    intrinsic[0, 2] = float(lines[2].split()[-1])
    intrinsic[1, 2] = float(lines[3].split()[-1])
    return intrinsic
        

def init_fn(i):
    return np.random.seed(torch.initial_seed() % 2 ** 32 - i)
        
class RGBVotingModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = ModelRaw(self.cfg)
        np.random.seed(int(round(time.time() * 1000)) % (2**32 - 1))
        # self.images = []
        self.root = sorted(glob(os.path.join('data/{}'.format(self.cfg.model_name.name), '*/')))[-1]
        
            
        # assert len(self.images) > 0
        
    def train_dataloader(self):
        return DataLoader(
            OnePoseRawDataset(self.cfg),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.num_workers,
            worker_init_fn=init_fn
        )
        
    def val_dataloader(self):
        class DummyDataset(Dataset):
            def __init__(self) -> None:
                super().__init__()
            
            def __len__(self):
                return 1
            
            def __getitem__(self, index):
                return 1
        return DataLoader(
            DummyDataset(),
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=1,
            worker_init_fn=init_fn
        )
        
    def training_step(self, batch, batch_idx):
        loss_dict = self.model(batch)
        loss = torch.sum(torch.stack(list(loss_dict.values())))
        for k, v in loss_dict.items():
            self.log(k, v.item(), prog_bar=True, on_step=True, on_epoch=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        idx = np.random.randint(len(glob(os.path.join(self.root, 'color_full/*.png'))))
        pose = np.loadtxt(Path(self.root) / 'poses_ba' / f'{idx}.txt')
        rgb = cv2.imread(os.path.join(self.root, f'color_full/{idx}.png'))[..., ::-1].copy()
        intrinsics = load_intrinsic(Path(self.root) / 'intrinsics.txt')
        
        inputs = {
            'rgbs': torch.from_numpy((np.moveaxis(rgb, -1, 0) / 255.).astype(np.float32)[None]).cuda(),
            'intrinsics': torch.from_numpy(intrinsics.astype(np.float32)[None]).cuda(),
            'poses': torch.from_numpy(pose.astype(np.float32))[None].cuda()
        }
        self.model.predict(inputs)
        return 
    
    def validation_epoch_end(self, outputs) -> None:
        pass
    
    def forward(self, image, intrinsic):
        return self.model.inference(image, intrinsic)
        
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=3e-4, weight_decay=1e-4)
        # return AdamW(self.parameters(), lr=1e-3, weight_decay=0)
    
@hydra.main(config_path='configs', config_name='cls', version_base='1.2')
def main(cfg):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=output_dir)
    callback = ModelCheckpoint(save_last=True, every_n_epochs=10, save_top_k=-1)
    pl_module = RGBVotingModule(cfg)
    trainer = pl.Trainer(max_epochs=100, gpus=[0], num_sanity_val_steps=0, logger=tb_logger, callbacks=[callback]) # check_val_every_n_epoch=10)
    trainer.fit(pl_module)
    
if __name__ == '__main__':
    main()