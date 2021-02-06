import json
import os

import torch
import torch.nn as nn
from torchvision import transforms, models
import timm

from ..module import Similarity, ImageEncoder

MODEL_CLASSES = {
    "ResNet-18": models.resnet18,
    "ResNet-34": models.resnet34,
    "ResNet-50": models.resnet50,
    "ResNet-101": models.resnet101,
    "ResNet-152": models.resnet152
}

def load_OML_ImageFolder_models(model_dir):
    embedder_statedict = torch.load(
        os.path.join(model_dir, "embedder.pth"), map_location="cpu"
    )
    trunk_statedict = torch.load(
        os.path.join(model_dir, "trunk.pth"), map_location="cpu"
    )
    with open(os.path.join(model_dir, "conf.json")) as h:
        CONF, PARAMS = json.load(h)
    
    if CONF["trunk_model"] in MODEL_CLASSES:
        trunk = MODEL_CLASSES[CONF["trunk_model"]](pretrained=True)
        trunk_output_size = trunk.fc.in_features
        trunk.fc = nn.Identity()
    elif CONF["trunk_model"].startswith("timm:"):
        _model_name = CONF["trunk_model"][5:]
        trunk = timm.create_model(_model_name, pretrained=True)
        trunk.reset_classifier(0)
        trunk_output_size = trunk.num_features

    embedder = nn.Sequential(
        nn.Linear(trunk_output_size, CONF["dim"]),
        nn.Dropout(PARAMS["p_dropout"])
    )

    trunk.load_state_dict(trunk_statedict)
    embedder.load_state_dict(embedder_statedict)
    
    return trunk, embedder

class OML_ImageFolder_Pretrained(nn.Module):
    def __init__(self, model_dir):
        super().__init__()

        trunk, embedder = load_OML_ImageFolder_models(model_dir)        

        self.trunk = trunk
        self.embedder = embedder
    
    def forward(self, x):
        x = self.trunk(x)
        x = self.embedder(x)
        x = x / torch.norm(x, dim=1, keepdim=True)
        return x

def create_OML_ImageFolder_Encoder(model_dir):
    with open(os.path.join(model_dir, "conf.json")) as h:
        CONF, PARAMS = json.load(h)

    input_size = CONF["input_size"]
    transform = [
        transforms.Resize((input_size,input_size)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    transform = transforms.Compose(transform)

    return ImageEncoder(
        OML_ImageFolder_Pretrained(model_dir),
        transform
    )