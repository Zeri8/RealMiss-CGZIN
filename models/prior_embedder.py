import torch
import torch.nn as nn

class ClinicalPriorEmbedder(nn.Module):
    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.mask_proj = nn.Linear(4, embed_dim)        # missing_mask
        self.mode_embed = nn.Embedding(4, embed_dim)    # 4种临床模式

        def forward(self, missing_mask: torch.Tensor, mode_id: torch.Tensor):
            """
            Args:
                missing_mask: [B, 4], float32, 1=存在, 0=缺失
                mode_id: [B], long, 0~3
            Returns:
                prior_vec: [B, embed_dim]
            """
            mask_vec = self.mask_proj(missing_mask.float())  # [B,16]
            mode_vec = self.mode_embed(mode_id)  # [B,16]
            prior_vec = mask_vec + mode_vec  # 融合
            return prior_vec