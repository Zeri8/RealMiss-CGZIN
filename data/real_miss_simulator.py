import random
import json
from pathlib import Path

CLINICAL_MODES = [
    'allergy_no_t1ce',   # 0    过敏无t1ce
    'old_scanner',       # 1
    'emergency',         # 2
    'full'               # 3
]

MODALITY_ORDER = ['T1', 'T1ce', 'T2', 'FLAIR']


def real_miss_simulator(case_id: str, phase: str = 'train'):
    """
    模拟临床真实缺失模式
    """
    if phase == 'test':
        random.seed(hash(case_id) % 1000)  # 确定性
    else:
        random.seed()

    r = random.random()
    if r < 0.40:  # 40% 过敏 → 无 T1ce
        mask = [1, 0, 1, 1]
        mode_id = 0
    elif r < 0.65:  # 25% 老设备 → 无 FLAIR
        mask = [1, 1, 1, 0]
        mode_id = 1
    elif r < 0.85:  # 20% 急诊 → 仅 T1+T2
        mask = [1, 0, 1, 0]
        mode_id = 2
    else:  # 15% 常规
        mask = [1, 1, 1, 1]
        mode_id = 3

    return {
        'missing_mask': mask,
        'mode_id': mode_id,
        'mode_name': CLINICAL_MODES[mode_id],
        'available_modalities': [MODALITY_ORDER[i] for i, v in enumerate(mask) if v == 1]
    }


# 测试
if __name__ == "__main__":
    result = real_miss_simulator("case_001")
    print(json.dumps(result, indent=2))

