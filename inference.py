import os
from PIL import Image
import io
import json
import pickle
import numpy as np
from smdebug import modes
import smdebug.pytorch as smd

from sagemaker_inference import content_types, decoder

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def model_fn(model_dir):

    logger.info("In model_fn. Model directory is -")
    logger.info(model_dir)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model = torch.jit.load(f)
    return model


def predict_fn(input_data, model):
    
    model.eval()
    with torch.no_grad():
        return model(input_data)