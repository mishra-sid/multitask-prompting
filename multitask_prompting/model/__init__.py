from multitask_prompting.model.mlm import MLM
from multitask_prompting.model.warp import WARP
from multitask_prompting.model.warp_share import WARPShare

MODEL_CLASSES = {
    'classification': {
        'mlm': MLM,
        'warp': WARPShare, 
    }
}

def get_model(task, model):
    return MODEL_CLASSES[task][model]