import torch
from torch import nn

"""
backpropagation on a very simple neural network
"""
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

loss.backward()  # compute gradient for all variables that were assigned requires_grad=True
print(w.grad)
print(b.grad)

z = torch.matmul(x, w) + b  # computation allowing for automatic differentiation
print(z.requires_grad)

with torch.no_grad():  # disabling automatic differentiation
    # (already trained models and other scenarios that only need forward passes, frozen parameters for finetuning)
    z = torch.matmul(x, w) + b
print(z.requires_grad)

z = torch.matmul(x,w)+b
z_det = z.detach()  # alternative method of disabling gradient tracking
print(z_det.requires_grad)

"""
backpropagation for a multidimensional output (Jacobian instead of simple gradient)
"""
inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nFirst call\n{inp.grad}")
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"Call after zeroing gradients\n{inp.grad}")
