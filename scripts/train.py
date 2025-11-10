# scripts/train.py
import torch
import yaml
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.mmformer_cgzin import CGZINImputer
from data.brats_dataset import BraTSRealMissDataset
from utils.loss import DiceCELoss, KLLoss

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/train_config.yaml')
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

os.makedirs(cfg['paths']['ckpt_dir'], exist_ok=True)
writer = SummaryWriter(cfg['paths']['log_dir'])

device = 'cuda'
model = CGZINImputer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
dice_ce = DiceCELoss()
kl_loss = KLLoss()

train_dataset = BraTSRealMissDataset(cfg['data']['root'], 'train', tuple(cfg['data']['img_size']), curriculum_epoch=0, total_epochs=cfg['train']['epochs'])
train_loader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=4)

best_dice = 0.0
for epoch in range(cfg['train']['epochs']):
    train_dataset.curriculum_epoch = epoch + 1
    model.train()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        x_dict = {k: v.to(device) for k, v in batch['x_dict'].items()}
        seg = batch['seg'].to(device).long()
        missing_mask = batch['missing_mask'].to(device)
        mode_id = batch['mode_id'].to(device)

        optimizer.zero_grad()
        output = model(x_dict, missing_mask, mode_id)

        loss_seg = dice_ce(output['seg'], seg)
        loss_kl = sum(kl_loss(F.log_softmax(p, 1), F.softmax(output['seg'].detach(), 1))
                      for i, p in enumerate(output['single_preds']) if missing_mask[0,i]==1) / missing_mask.sum()
        loss_mod = nn.MSELoss()(output['mod_prob'], missing_mask.float())
        loss = cfg['train']['dice_weight'] * loss_seg + cfg['train']['kl_weight'] * loss_kl + cfg['train']['mod_weight'] * loss_mod
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), os.path.join(cfg['paths']['ckpt_dir'], 'last.pth'))