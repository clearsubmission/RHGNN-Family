from models.rhgnn_v1a import RHGNNBase      # V1a: RHGNN-HLSTM
from models.rhgnn_v1b import RHGNNHgru      # V1b: RHGNN-HGRU
from models.rhgnn_v2  import RHGNNFreqAware # V2:  RHGNN-FA
from models.rhgnn_v3  import RHGNNMemory    # V3:  RHGNN-MA
from models.rhgnn_v4  import RHGNNCA        # V4:  RHGNN-CA (full model)
from models.rhgnn_v5_pure import RHGNNFH    # V5:  RHGNN-FH (geometry ablation)

MODEL_REGISTRY = {
    'rhgnn_v1a': RHGNNBase,
    'rhgnn_v1b': RHGNNHgru,
    'rhgnn_v2':  RHGNNFreqAware,
    'rhgnn_v3':  RHGNNMemory,
    'rhgnn_v4':  RHGNNCA,
    'rhgnn_v5':  RHGNNFH,
    # Aliases
    'rhgnn_hlstm': RHGNNBase,
    'rhgnn_hgru':  RHGNNHgru,
    'rhgnn_fa':    RHGNNFreqAware,
    'rhgnn_ma':    RHGNNMemory,
    'rhgnn_ca':    RHGNNCA,
    'rhgnn_fh':    RHGNNFH,
}

def get_model(name):
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. "
                         f"Choose from: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]
