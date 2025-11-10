# scripts/inference_vis.py
import torch
import argparse
from models.mmformer_cgzin import CGZINImputer
from data.brats_dataset import BraTSRealMissDataset
from utils.visualize import save_vis

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='checkpoints/last.pth')
parser.add_argument('--idx', type=int, default=0)
args = parser.parse_args()

model = CGZINImputer().cuda()
model.load_state_dict(torch.load(args.ckpt))
model.eval()

ds = BraTSRealMissDataset('data/raw', 'val')
item = ds[args.idx]
x = {k: v.unsqueeze(0).cuda() for k,v in item['x_dict'].items()}
with torch.no_grad():
    out = model(x, item['missing_mask'].unsqueeze(0).cuda(), item['mode_id'].unsqueeze(0).cuda())

save_vis(out, x['T1CE'], item['seg'], item['case_id'])