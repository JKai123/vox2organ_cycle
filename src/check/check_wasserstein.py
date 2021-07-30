
""" Put module information here """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import time

import torch
from geomloss import SamplesLoss

x = torch.randn(40000, 3, requires_grad=True).cuda()
y = torch.randn(40000, 3).cuda()

loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)

tic = time.perf_counter()
L = loss(x, y)
g_x, = torch.autograd.grad(L, [x])
toc = time.perf_counter()

time_elapsed = toc - tic

print("Time for forward and backward computation of Wasserstein distance"
      f" between two pointclouds of 40000 vertices: {time_elapsed} s")
