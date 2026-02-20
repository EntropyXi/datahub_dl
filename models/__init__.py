from models.cnn import SimpleCNN
from models.resnet import ResNet
from models.vae import ConvVAE

MODEL_REGISTRY = {
    "resnet": ResNet,
    "vae": ConvVAE,
    "cnn": SimpleCNN, 
}