
import torch
import torchvision
from torch import nn

def create_model(num_classes:int=2,
                 seed:int=42):
  weights=torchvision.models.ResNet50_Weights.DEFAULT
  transforms=weights.transforms
  model=torchvision.models.resnet50(weights=weights)

  for param in model.parameters():
    param.requires_grad=False

  torch.manual_seed(42)
  model.fc= torch.nn.Sequential(
    torch.nn.Linear(2048,1000),
    torch.nn.ReLU(),
    torch.nn.Linear(1000,500),
    torch.nn.Dropout(),
    torch.nn.Linear(in_features=500,
                    out_features=output_shape,
                    bias=True)

    )

  return model,transforms

