from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt


class Net(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28 * 28, 2 ** 9)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2 ** 9, 2 ** 9)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(2 ** 9, 10)

        return

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)

        return x


def training(
        model: Net,
        data_train: DataLoader,
        data_test: DataLoader,
        loss_function: functional,
        goal_accuracy: float,
        learning_rate: float = 1e-3
) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    accuracy = loop_test(model, data_test)
    epoch = 0

    print("-----------------------------------")
    while accuracy < goal_accuracy:
        print(f"Epoch {epoch:>3d} | Starting Accuracy {100 * accuracy:>4.1f}%")
        print("-----------------------------------")
        epoch += 1
        loop_train(model, data_train, loss_function, optimizer)
        print("-----------------------------------")
        accuracy = loop_test(model, data_test)
    print(f"Epochs {epoch:>3d} | Final Accuracy {100 * accuracy:>4.1f}%")

    return


def loop_train(model: Net, data_train: DataLoader, loss_function: functional, optimizer) -> None:
    size = len(data_train.dataset)

    for batch, (X, y) in enumerate(data_train):
        prediction = model(X)
        loss = loss_function(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"   loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    return


def loop_test(model: Net, data_test: DataLoader) -> float:
    size = len(data_test.dataset)
    correct = 0

    with torch.no_grad():
        for X, y in data_test:
            prediction = model(X)
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

    correct /= size

    return correct


# toy = Net()
# dataset_train = datasets.MNIST(
#     root='data',
#     train=True,
#     transform=ToTensor(),
#     download=True,
# )
# dataset_test = datasets.MNIST(
#     root='data',
#     train=False,
#     transform=ToTensor(),
#     download=True
# )
# dataloader_train = DataLoader(dataset_train, batch_size=60, shuffle=True)
# dataloader_test = DataLoader(dataset_test, batch_size=100, shuffle=True)
# loss_function = nn.CrossEntropyLoss()
# goal_accuracy_test = .8
# accuracy = .0
# training(toy, dataloader_train, dataloader_test, loss_function, goal_accuracy_test)
# torch.save(toy, 'toy.pt')

toy = torch.load('toy.pt')
dataset_test = datasets.MNIST(
    root='data',
    train=False,
    transform=ToTensor(),
    download=True
)

assignments = torch.zeros(10, 10)
for image, label in dataset_test:
    assignments[int(label), int(toy(image).argmax(axis=1))] += 1
    assignments[int(toy(image).argmax(axis=1)), int(label)] += 1  # symmetrize
figure = plt.figure(figsize=(4, 4))
plt.axis('off')
plt.imshow(assignments, cmap='binary')
plt.show()

mapping = [1, 0, 0, 1, 2, 1, 0, 2, 1, 2]

assignments_new = torch.zeros(3, 3)

for image, label in dataset_test:
    assignments_new[mapping[int(label)], mapping[int(toy(image).argmax(axis=1))]] += 1
    assignments_new[mapping[int(toy(image).argmax(axis=1))], mapping[int(label)]] += 1  # symmetrize
figure = plt.figure(figsize=(4, 4))
plt.axis('off')
plt.imshow(assignments_new, cmap='binary')
plt.show()
print(assignments_new)


class NetRestrained(nn.Module):

    def __init__(self, basis: Net):
        super(NetRestrained, self).__init__()
        self.flatten = basis.flatten
        self.linear1 = basis.linear1
        self.relu1 = basis.relu1
        self.linear2 = basis.linear2
        self.relu2 = basis.relu2
        self.linear3 = basis.linear3
        self.linear4 = nn.Linear(10, 3)
        a = torch.tensor([
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
        ])
        self.linear4.weight = nn.Parameter(a)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.linear4(x)

        return x


toy = NetRestrained(toy)
assignments_new_2 = torch.zeros(3, 3)

for image, label in dataset_test:
    assignments_new_2[mapping[int(label)], int(toy(image).argmax(axis=1))] += 1
    assignments_new_2[int(toy(image).argmax(axis=1)), mapping[int(label)]] += 1  # symmetrize
figure = plt.figure(figsize=(4, 4))
plt.axis('off')
plt.imshow(assignments_new_2, cmap='binary')
plt.show()
print(assignments_new_2)
