import torch
import torch.nn as nn
from .mmformer_base import MMFormerBase
from .prior_embedder import ClinicalPriorEmbedder
from .multitask_head import MultiTaskHead
from utils.suggestion import generate_suggestion

class CGZINImputer(nn.Module):
    def __init__(self, img_size=(128,128,128), embed_dim=16):
        super().__init__()
        self.base = MMFormerBase(num_cls=4)
        self.base.is_training = True
        self.prior_embedder = ClinicalPriorEmbedder(embed_dim)
        self.prior_proj = nn.Linear(embed_dim, 512)
        self.multitask_head = MultiTaskHead()

    def forward(self, x_dict, missing_mask, mode_id):
        B = missing_mask.shape[0]
        mods = ['FLAIR', 'T1', 'T1CE', 'T2']
        x_list = []
        for i, mod in enumerate(mods):
            if mod in x_dict:
                x_list.append(x_dict[mod])
            else:
                x_list.append(torch.zeros(B, 1, *x_dict[mods[0]].shape[2:], device=x_dict[mods[0]].device))
        x = torch.cat(x_list, dim=1)
        mask = missing_mask.bool()

        fuse_pred, (p1, p2, p3, p4), deep_preds = self.base(x, mask)

        mod_prob = self.multitask_head(fuse_pred)
        suggestion = generate_suggestion(missing_mask[0], mod_prob[0])

        return {
            'seg': fuse_pred,
            'mod_prob': mod_prob,
            'suggestion': suggestion,
            'single_preds': (p1, p2, p3, p4),
            'deep_preds': deep_preds
        }