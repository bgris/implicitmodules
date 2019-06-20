import copy

import torch
import torch.optim
from .LBFGS import FullBatchLBFGS

from implicitmodules.torch.DeformationModules import CompoundModule, SilentLandmarks
from implicitmodules.torch.HamiltonianDynamic import Hamiltonian, shoot
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.Utilities.usefulfunctions import grid2vec, vec2grid


class ModelAtlas():
    def __init__(self, attachement):
        self.__attachement = attachement

    @property
    def attachement(self):
        return self.__attachement

    def compute(self, targets):
        raise NotImplementedError

    def __call__(self, reverse=False):
        raise NotImplementedError

    def transform_target(self, target):
        return target

    def fit(self, targets, lr=1e-3, l=10., max_iter=100, tol=1e-7, log_interval=10):
        #transformed_target = self.transform_target(target)

        optimizer = torch.optim.LBFGS(self.parameters, lr=lr, max_iter=4)
        optim = FullBatchLBFGS(self.parameters, lr=step_length, history_size=500, line_search='Wolfe')
        self.nit = -1
        self.break_loop = False
        costs = []

        def closure():
            self.nit += 1
            optimizer.zero_grad()
            if self.precompute_cb is not None:
                self.precompute_cb(self.parameters)
            self.compute(self.targets)
            attach = l*self.attach
            cost = attach + self.deformation_cost
            if(self.nit%log_interval == 0):
                print("It: %d, deformation cost: %.6f, attach: %.6f. Total cost: %.6f" % (self.nit, self.deformation_cost.detach().numpy(), attach.detach().numpy(), cost.detach().numpy()))

            costs.append(cost.item())

            if(len(costs) > 1 and abs(costs[-1] - costs[-2]) < tol) or self.nit >= max_iter:
                self.break_loop = True
            else:
                cost.backward(retain_graph=True)
            return cost

        for i in range(0, max_iter):
            optimizer.step(closure)

            if(self.break_loop):
                break

        print("End of the optimisation process.")
        return costs


class ModelCompoundAtlas:
    def __init__(self, modules, fixed, targets, attachement, precompute_callback, precompute_cb=None, other_parameters=[]):
        #super().__init__(attachement)
        self.__nb_pop = len(targets)
        self.__targets = targets
        self.__modules = modules
        self.__attachement = attachement
        self.__precompute_callback = precompute_callback
        self.__fixed = fixed
        self.__precompute_cb = precompute_cb
        
        self.__init_manifolds = [CompoundModule(self.__modules).manifold.copy() for  k in range(self.__nb_pop)]
        self.__init_parameters = copy.copy(other_parameters)
        self.__init_other_parameters = []
        for p in other_parameters:
            #self.__init_other_parameters.append(p.detach().clone().requires_grad_())
            self.__init_other_parameters.append(p)

        self.compute_parameters()
        

    @property
    def nb_pop(self):
        return self.__nb_pop

    @property
    def targets(self):
        return self.__targets

    @property
    def modules(self):
        return self.__modules

    @property
    def fixed(self):
        return self.__fixed

    @property
    def attachement(self):
        return self.__attachement

    @property
    def precompute_callback(self):
        return self.__precompute_callback

    @property
    def precompute_cb(self):
        return self.__precompute_cb

    @property
    def init_manifolds(self):
        return self.__init_manifolds

    @property
    def init_parameters(self):
        return self.__init_parameters

    @property
    def parameters(self):
        return self.__parameters


    def compute_parameters(self):    
        self.__parameters = []

        # initial GDs, taken from the 1st element
        for i in range(len(self.__modules)):
            if(not self.__fixed[i]):
                self.__parameters.extend(self.__init_manifolds[0][i].unroll_gd())
                
        # Momenta
        for k in range(self.__nb_pop):            
            for i in range(len(self.__modules)):
                self.__parameters.extend(self.__init_manifolds[k][i].unroll_cotan())

        self.__parameters.extend(self.__init_other_parameters)


    def compute_deformation_grid(self, k, grid_origin, grid_size, grid_resolution, it=2, intermediate=False):
        x, y = torch.meshgrid([
            torch.linspace(grid_origin[0], grid_origin[0]+grid_size[0], grid_resolution[0]),
            torch.linspace(grid_origin[1], grid_origin[1]+grid_size[1], grid_resolution[1])])

        gridpos = grid2vec(x, y)

        grid_landmarks = Landmarks(2, gridpos.shape[0], gd=gridpos.view(-1))
        grid_silent = SilentLandmarks(grid_landmarks)
        compound = CompoundModule(self.modules)
        compound.manifold.fill(self.init_manifolds[k])

        intermediate_states, _ = shoot(Hamiltonian([grid_silent, *compound]), 10, "torch_euler")

        return [vec2grid(inter[0].gd.detach().view(-1, 2), grid_resolution[0], grid_resolution[1]) for inter in intermediate_states]


class ModelCompoundWithPointsAtlas(ModelCompoundAtlas):
    def __init__(self, source, module_list, fixed_template, fixed, targets, attachement, precompute_callback=None, other_parameters=[]):
        if isinstance(source, list):
            self.__compound_fit = True
            self.__compound_fit_size = len(source)
            self.alpha = []
            for i in range(self.__compound_fit_size):
                #self.alpha.insert(i, source[i][1])
                module_list.insert(i, SilentLandmarks(Landmarks(2, source[i][0].shape[0], gd=source[i][0].view(-1).requires_grad_())))
                fixed.insert(i, fixed_template)

        else:
            self.__compound_fit = False
            #self.alpha = source[1]
            module_list.insert(0, SilentLandmarks(Landmarks(2, source[0].shape[0], gd=source[0].view(-1).requires_grad_())))
            fixed.insert(0, fixed_template)

        super().__init__(module_list, fixed, targets, attachement, precompute_callback, other_parameters=other_parameters)

    def compute(self, targets):
        compound = CompoundModule(self.modules)
        self.shot_manifold = []
        attachs = []
        def_costs = []
        for k in range(self.nb_pop):
            self.init_manifolds[k].fill_gd(self.init_manifolds[0].gd)
            compound.manifold.fill(self.init_manifolds[k])
            h = Hamiltonian(compound)
            shoot(h, 10, "torch_euler")
            self.shot_manifold.append(compound.manifold.copy())
            def_costs.append(compound.cost())

            if self.__compound_fit:
                self.__shot_points = []
                attach_list = []
                for i in range(self.__compound_fit_size):
                    self.__shot_points.append(compound[i].manifold.gd.view(-1, 2))
                    attach_list.append(self.attachement[i]((compound[i].manifold.gd.view(-1, 2), 0), (self.targets[k][i], 0)))
                attachs.append(sum(attach_list))
            else:
                self.__shot_points = compound[0].manifold.gd.view(-1, 2)
                attachs.append(self.attachement((self.__shot_points, 0), (self.targets[k], 0)))
                
        self.deformation_cost = sum(def_costs)
        self.attach = sum(attachs)
        return self.deformation_cost, self.attach
    def __call__(self):
        if self.__compound_fit:
            return list(self.__shot_points)
        else:
            return self.__shot_points

