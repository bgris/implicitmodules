import os.path
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import time
import torch

import implicitmodules.torch as im

parser = argparse.ArgumentParser()
parser.add_argument("device", type=str, help="Device on which to perform computations.", choices=['cpu', 'cuda'])
parser.add_argument("method", type=str, help="Numerical sheme used for shooting.")
parser.add_argument("nb_silent", type=int, help="Number of silent points to shoot.")
parser.add_argument("nb_order0", type=int, help="Number of order 0 points to shoot.")
parser.add_argument("nb_order1", type=int, help="Number of order 1 points to shoot.")

parser.add_argument("-it", type=int, help="Number of iterations for the shooting.", default=10)
parser.add_argument("-loops", type=int, help="Number of shooting to get a mean shooting time.", default=10)
args = parser.parse_args()


def simple_shooting(method, it, device, points):
    dim = 2
    nb_pts_silent = int(points[0])
    nb_pts_order0 = int(points[1])
    nb_pts_order1 = int(points[2])

    pts_silent = 100.*torch.rand(nb_pts_silent, dim)
    pts_order0 = 100.*torch.rand(nb_pts_order0, dim)
    pts_order1 = 100.*torch.rand(nb_pts_order1, dim)
    p_pts_order1 = torch.rand(nb_pts_order1, dim)

    nu = 0.001
    coeff = 1.
    sigma = 1.5

    C = torch.rand(nb_pts_order1, 2, 1)
    R = torch.rand(nb_pts_order1, 2, 2)
    p_R = torch.rand(nb_pts_order1, 2, 2)

    silent = im.DeformationModules.SilentLandmarks(im.Manifolds.Landmarks(dim, nb_pts_silent, gd=pts_silent.view(-1).requires_grad_()))
    order0 = im.DeformationModules.Translations(im.Manifolds.Landmarks(dim, nb_pts_order0, gd=pts_order0.view(-1).requires_grad_()), sigma)
    order1 = im.DeformationModules.ImplicitModule1(im.Manifolds.Stiefel(dim, nb_pts_order1, gd=(pts_order1.view(-1).requires_grad_(), R.view(-1).requires_grad_())), C, sigma, nu, coeff)

    compound = im.DeformationModules.CompoundModule([silent, order0, order1])
    compound.move_to(device)
    start = time.time()
    im.HamiltonianDynamic.shooting.shoot(im.HamiltonianDynamic.Hamiltonian(compound), it=it, method=method)
    elapsed = time.time() - start
    compound.move_to('cpu')

    return elapsed


def test_method(method, it, loops, device, points):
    time_shooting = []
    time_back = []
    for i in range(loops):
        elapsed = simple_shooting(method, it, device, points)
        time_shooting.append(elapsed)

    return sum(time_shooting)/loops


def method_summary(method, it, loops, device, points):
    avg_shoot = test_method(method, it, loops, device, points)

    print("For method %s, on device %s, with (%i, %i, %i), average shooting time: %5.4f s." % (method, device, points[0], points[1], points[2], avg_shoot))


torch.set_printoptions(precision=4)

method_summary(args.method, args.it, args.loops, args.device,
               (args.nb_silent, args.nb_order0, args.nb_order1))

