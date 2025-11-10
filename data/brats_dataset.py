import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import nibabel as nib
from scipy.ndimage import zoom
from typing import Dict, Tuple, Any

MODALITY_ORDER = ['FLAIR', 'T1', 'T1CE', 'T2']

class BraTSRealMissDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        phase: str = 'train',
        img_size: Tuple[int, int, int] = (128, 128, 128),
        cache: bool = True,
        curriculum_epoch: int = 0,
        total_epochs: int = 100
    ):
        self.data_dir = Path(data_dir)
        self.phase = phase
        self.img_size = img_size
        self.cache = cache
        self.cached_data: Dict[str, Any] = {}
        self.curriculum_epoch = curriculum_epoch
        self.total_epochs = total_epochs

        cases_dir = self.data_dir / 'MICCAI_BraTS2020_TrainingData'
        if not cases_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {cases_dir}")

        self.cases = sorted([
            d.name for d in cases_dir.iterdir()
            if d.is_dir() and d.name.startswith('BraTS20')
        ])

        split_idx = int(len(self.cases) * 0.8)
        if phase == 'train':
            self.cases = self.cases[:split_idx]
        else:
            self.cases = self.cases[split_idx:]

    def _is_curriculum_active(self) -> bool:
        return self.curriculum_epoch > 0

    def preprocess(self, vol: np.ndarray) -> np.ndarray:
        vol = vol.astype(np.float32)
        vol = (vol - vol.mean()) / (vol.std() + 1e-8)
        return self._resize(vol, order=1)

    def preprocess_seg(self, seg: np.ndarray) -> np.ndarray:
        seg = seg.copy().astype(np.uint8)
        seg[seg == 4] = 3
        return self._resize(seg, order=0)

    def _resize(self, data: np.ndarray, order: int = 1) -> np.ndarray:
        if data.shape == self.img_size:
            return data
        factors = [s / f for s, f in zip(self.img_size, data.shape)]
        return zoom(data, factors, order=order)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case_id = self.cases[idx]

        if self.cache and case_id in self.cached_data:
            return self.cached_data[case_id]

        case_dir = self.data_dir / 'MICCAI_BraTS2020_TrainingData' / case_id
        modalities = {}
        for mod in MODALITY_ORDER:
            path = case_dir / f'{case_id}_{mod.lower()}.nii.gz'
            if not path.exists():
                raise FileNotFoundError(f"Missing file: {path}")
            img = nib.load(path).get_fdata()
            img = self.preprocess(img)
            modalities[mod] = img

        seg_path = case_dir / f'{case_id}_seg.nii.gz'
        if not seg_path.exists():
            raise FileNotFoundError(f"Missing seg: {seg_path}")
        seg = nib.load(seg_path).get_fdata().astype(np.uint8)
        seg = self.preprocess_seg(seg)

        from .real_miss_simulator import real_miss_simulator
        miss_info = real_miss_simulator(
            case_id=case_id,
            phase=self.phase,
            curriculum_epoch=self.curriculum_epoch,
            total_epochs=self.total_epochs
        )
        missing_mask = torch.tensor(miss_info['missing_mask'], dtype=torch.float32)
        mode_id = torch.tensor(miss_info['mode_id'], dtype=torch.long)

        if self.phase == 'train' and self._is_curriculum_active():
            if self.curriculum_epoch < self.total_epochs * 0.4:
                if random.random() < 0.3:
                    missing_mask[3] = 0
                    mode_id = torch.tensor(4)

        x_dict = {}
        for i, mod in enumerate(MODALITY_ORDER):
            if missing_mask[i] == 1:
                x_dict[mod] = torch.from_numpy(modalities[mod]).unsqueeze(0).float()
            else:
                x_dict[mod] = torch.zeros(1, *self.img_size, dtype=torch.float32)

        item = {
            'x_dict': x_dict,
            'missing_mask': missing_mask,
            'mode_id': mode_id,
            'seg': torch.from_numpy(seg),
            'case_id': case_id
        }

        if self.cache:
            self.cached_data[case_id] = item

        return item