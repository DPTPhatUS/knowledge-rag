import torch

import torch.nn as nn
import torch.optim as optim


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

inputs = torch.randn(5, 10)
targets = torch.randn(5, 1)

outputs = model(inputs)
loss = criterion(outputs, targets)
print(f"Initial Loss: {loss.item()}")

optimizer.zero_grad()
loss.backward()
optimizer.step()

outputs = model(inputs)
loss = criterion(outputs, targets)
print(f"Loss after one step: {loss.item()}")
