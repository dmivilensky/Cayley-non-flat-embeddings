#!/usr/bin/env python3

import torch
import numpy as np

x = torch.Tensor(np.array([1, 2, 3, 4]))
x.requires_grad = True
y = torch.acosh(x)


def f(y):
    return torch.linalg.norm(y - torch.Tensor(np.array([1, 0, 1, 1])))

loss = f(y)
loss.backward()

print(x.grad.data)