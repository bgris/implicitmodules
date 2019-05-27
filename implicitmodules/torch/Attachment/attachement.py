from collections import Iterable

import geomloss
import torch

from implicitmodules.torch.Kernels.kernels import distances, scal
from implicitmodules.torch.Utilities.usefulfunctions import close_shape


class Attachement:
    def __init__(self):
        pass

    def __call__(self, x, y):
        return self.loss(x, y)

    def loss(self, x, y):
        raise NotImplementedError

class EnergyAttachement(Attachement):
    """Energy Distance between two sampled probability measures."""
    def __init__(self):
        super().__init__()

    def loss(self, x, y):
        x_i, a_i = x
        y_j, b_j = y
        K_xx = -distances(x_i, x_i)
        K_xy = -distances(x_i, y_j)
        K_yy = -distances(y_j, y_j)
        return .5*scal(a_i, torch.mm(K_xx, a_i.view(-1, 1))) - scal(a_i, torch.mm(K_xy, b_j.view(-1, 1))) + .5*scal(b_j, torch.mm(K_yy, b_j.view(-1, 1)))

class L2NormAttachement(Attachement):
    """L2 norm distance between two measures."""
    def __init__(self):
        super().__init__()

    def loss(self, x, y):
        return torch.dist(a, b)


class VarifoldAttachement(Attachement):
    def __init__(self, sigmas):
        assert isinstance(sigmas, Iterable)
        super().__init__()
        self.__sigmas = sigmas

    @property
    def sigmas(self):
        return self.__sigmas

    def __cost_varifold(self, x, y, sigma):
        def dot_varifold(x, y, sigma):
            device = x.device
            cx, cy = close_shape(x), close_shape(y)
            nx, ny = x.shape[0], y.shape[0]

            vx, vy = cx[1:nx + 1, :] - x, cy[1:ny + 1, :] - y
            mx, my = (cx[1:nx + 1, :] + x) / 2, (cy[1:ny + 1, :] + y) / 2

            xy = torch.tensordot(torch.transpose(torch.tensordot(mx, my, dims=0), 1, 2), torch.eye(2, device=device))

            d2 = torch.sum(mx * mx, dim=1).reshape(nx, 1).repeat(1, ny) + torch.sum(my * my, dim=1).repeat(nx, 1) - 2 * xy

            kxy = torch.exp(-d2 / (2 * sigma ** 2))

            vxvy = torch.tensordot(torch.transpose(torch.tensordot(vx, vy, dims=0), 1, 2), torch.eye(2, device=device)) ** 2

            nvx = torch.sqrt(torch.sum(vx * vx, dim=1))
            nvy = torch.sqrt(torch.sum(vy * vy, dim=1))

            mask = vxvy > 0

            return torch.sum(kxy[mask] * vxvy[mask] / (torch.tensordot(nvx, nvy, dims=0)[mask]))

        return dot_varifold(x, x, sigma) + dot_varifold(y, y, sigma) - 2 * dot_varifold(x, y, sigma)

    def loss(self, x, y):
        return sum([self.__cost_varifold(x[0], y[0], s) for s in self.__sigmas])


class GeomlossAttachement(Attachement):
    def __init__(self, **kwargs):
        super().__init__()
        self.__geomloss = geomloss.SamplesLoss(**kwargs)

    def loss(self, x, y):
        return self.__geomloss.loss(x, y)


class PointwiseDistanceAttachement(Attachement):
    def __init__(self, norm='L1'):
        super().__init__()

    def loss(self, x, y):
        return torch.sum(torch.norm(x[0]-y[0], dim=1))
