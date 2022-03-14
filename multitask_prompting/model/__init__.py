from multitask_prompting.model.bert import BERT
from multitask_prompting.model.warp import WARP

MODEL_CLASSES = {
    'classification': {
        'bert': BERT,
        'warp': WARP, 
    }
}

def get_model(task, model):
    return MODEL_CLASSES[task][model]