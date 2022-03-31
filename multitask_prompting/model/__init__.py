from multitask_prompting.model.mlm import MLM
from multitask_prompting.model.warp import WARP

MODEL_CLASSES = {
    'classification': {
        'mlm': MLM,
        'warp': WARP, 
    }
}

def get_model(task, model):
    return MODEL_CLASSES[task][model]