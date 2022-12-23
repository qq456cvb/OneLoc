from pathlib import Path
import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset
from dataset import DemoDataset
from torch.optim import Adam, SGD, AdamW
import hydra
import cv2
from model import Model
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
        self.model = Model(self.cfg)
        np.random.seed(int(round(time.time() * 1000)) % (2**32 - 1))
        
        vidcap = cv2.VideoCapture('data/test.mp4')
        self.imgs = []
        while True:  
            success, image = vidcap.read()
            if not success:
                break
            self.imgs.append(image[..., ::-1])
        
    def train_dataloader(self):
        return DataLoader(
            DemoDataset(self.cfg),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.num_workers,
            worker_init_fn=init_fn
        )
        
    def training_step(self, batch, batch_idx):
        loss_dict = self.model(batch)
        loss = torch.sum(torch.stack(list(loss_dict.values())))
        for k, v in loss_dict.items():
            self.log(k, v.item(), prog_bar=True, on_step=True, on_epoch=False)
        return loss
    
    def training_epoch_end(self, outputs) -> None:
        if self.current_epoch % 1 == 0:
            # do evaluation
            rgb = self.imgs[np.random.randint(len(self.imgs))]
            self.model.predict(rgb)
    
    def forward(self, image, intrinsic):
        return self.model.inference(image, intrinsic)
        
    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=1e-3, weight_decay=0)
    
@hydra.main(config_path='.', config_name='config', version_base='1.2')
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