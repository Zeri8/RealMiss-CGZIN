
import torch.nn as nn

class ClinicalPriorEmbedder(nn.Module):
    def __init__(self, embed_dim=16):
        super().__init__()
        self.missing_emb = nn.Embedding(16, embed_dim // 2)
        self.mode_emb = nn.Embedding(5, embed_dim // 2)
        self.proj = nn.Linear(embed_dim, embed_dim)
    def forward(self, missing_mask, mode_id):
        miss_idx = (missing_mask[:,0]*8 + missing_mask[:,1]*4 + missing_mask[:,2]*2 + missing_mask[:,3]*1).long()
        miss_emb = self.missing_emb(miss_idx)
        mode_emb = self.mode_emb(mode_id)
        emb = torch.cat([miss_emb, mode_emb], dim=-1)
        return self.proj(emb)