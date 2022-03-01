from model.bert import BERT
from model.warp import WARP

MODEL_CLASSES = {
    'bert': BERT,
    'warp': WARP, 
}


def set_seed(seed):
    pass