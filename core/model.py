from timm.models import create_model
#Register Models
from models import *

# A wrapper for the timm model create function
# Handle the bug of loading pretrained weight in the new version of the pytorch
def createModel(model_name, pretrained=False,
        checkpoint_path='',
        scriptable=None,
        exportable=None,
        no_jit=None,
        **kwargs):
    model = create_model(model_name, pretrained=pretrained,
                         checkpoint_path=checkpoint_path, scriptable=scriptable,
                         exportable=exportable, no_jit=no_jit, **kwargs)

    return model

