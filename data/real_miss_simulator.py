import random
from typing import Dict

CLINICAL_SCENARIOS = {
    0: {'name': 'complete',        'mask': [1,1,1,1], 'risk': 'none'},
    1: {'name': 'no_t1ce',         'mask': [1,0,1,1], 'risk': 'contrast_allergy'},
    2: {'name': 'no_t1',           'mask': [0,1,1,1], 'risk': 'cost_control'},
    3: {'name': 'no_t2',           'mask': [1,1,0,1], 'risk': 'radiation_concern'},
    4: {'name': 'no_flair',        'mask': [1,1,1,0], 'risk': 'time_pressure'},
}

def real_miss_simulator(
    case_id: str,
    phase: str = 'train',
    curriculum_epoch: int = 0,
    total_epochs: int = 100
) -> Dict:
    if phase != 'train':
        return {'missing_mask': [1,1,1,1], 'mode_id': 0, 'scenario': 'complete'}

    mode_id = random.randint(0, 4)
    base_mask = CLINICAL_SCENARIOS[mode_id]['mask']

    if curriculum_epoch > 0 and curriculum_epoch < total_epochs * 0.4:
        if random.random() < 0.3:
            base_mask = [1,1,1,0]
            mode_id = 4

    return {
        'missing_mask': base_mask,
        'mode_id': mode_id,
        'scenario': CLINICAL_SCENARIOS[mode_id]['name']
    }