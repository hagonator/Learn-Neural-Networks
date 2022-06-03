import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted Class: {y_pred}")

"""
Going through the forward method step by step manually
"""
input_image = torch.rand(3, 28, 28)
print(input_image.size())

flatten = nn.Flatten()
flat_image = flatten(input_image)  # flatten array from 28x28 to 784
print(flat_image.size())

layer1 = nn.Linear(28 * 28, 20)
hidden1 = layer1(flat_image)  # apply linear transformation with weights defined by layer1
print(hidden1.size())

print(f"before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)  # apply non-linearity on the new representation
print(f"after ReLU: {hidden1}\n\n")

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10),
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)  # apply sequence of transformations (i.e. the model) onto the 'images'

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)  # find the class with highest probability (smoothly)

print(f"Model structure {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")  # print layer details
