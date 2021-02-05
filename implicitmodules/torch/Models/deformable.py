import os
from collections import Iterable
import pickle
import copy

import torch
from numpy import loadtxt, savetxt
import meshio

from implicitmodules.torch.HamiltonianDynamic import Hamiltonian, shoot
from implicitmodules.torch.DeformationModules import SilentBase, CompoundModule, SilentLandmarks
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.Utilities import deformed_intensities, AABB, load_greyscale_image, pixels2points
from implicitmodules.torch.MultiShape import MultiShapeHamiltonian
from implicitmodules.torch.MultiShape import MultiShape

class Deformable:
    def __init__(self, manifold, module_label=None):
        self.__silent_module = SilentBase(manifold, module_label)

    @property
    def silent_module(self):
        return self.__silent_module

    @property
    def geometry(self):
        raise NotImplementedError()

    @property
    def _has_backward(self):
        raise NotImplementedError()

    def _backward_module(self):
        raise NotImplementedError()

    def compute_deformed(self, modules, solver, it, costs=None, intermediates=None):
        raise NotImplementedError()

    def _to_deformed(self):
        raise NotImplementedError()


    def _to_deformed_new(self):
        raise NotImplementedError()


class DeformablePoints(Deformable):
    def __init__(self, points):
        super().__init__(Landmarks(points.shape[1], points.shape[0], gd=points))

    @classmethod
    def load_from_file(cls, filename, dtype=None, **kwargs):
        file_extension = os.path.split(filename)[1]
        if file_extension == '.csv':
            return cls.load_from_csv(filename, dtype=dtype, **kwargs)
        elif file_extension == '.pickle' or file_extension == '.pkl':
            return cls.load_from_pickle(filename, dtype=dtype)
        elif file_extension in meshio.extension_to_filetype.keys():
            return cls.load_from_mesh(filename, dtype=dtype)
        else:
            raise RuntimeError("DeformablePoints.load_from_file(): could not load file {filename}, unrecognised file extension!".format(filename=filename))

    @classmethod
    def load_from_csv(cls, filename, dtype=None, **kwargs):
        points = loadtxt(filename, **kwargs)
        return cls(torch.tensor(points, dtype=dtype))

    @classmethod
    def load_from_pickle(cls, filename, dtype=None):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                return cls(torch.tensor(data['points'], dtype=dtype))
            elif isinstance(data, Iterable):
                return cls(torch.tensor(data, dtype=dtype))
            else:
                raise RuntimeError("DeformablePoints.load_from_pickle(): could not infer point dataset from pickle {filename}".format(filename=filename))

    @classmethod
    def load_from_mesh(cls, filename, dtype=None):
        mesh = meshio.read(filename)
        return torch.tensor(mesh.points, dtype=dtype)

    @property
    def geometry(self):
        return (self.silent_module.manifold.gd,)

    @property
    def _has_backward(self):
        return False

    def save_to_file(self, filename, **kwargs):
        file_extension = os.path.split(filename)[1]
        if file_extension == '.csv':
            return self.save_to_csv(filename, **kwargs)
        elif file_extension == '.pickle' or file_extension == '.pkl':
            return self.save_to_pickle(filename, **kwargs)
        elif file_extension in meshio.extension_to_filetype.keys():
            return cls.save_to_mesh(filename, **kwargs)
        else:
            raise RuntimeError("DeformablePoints.load_from_file(): could not load file {filename}, unrecognised file extension!".format(filename=filename))

    def save_to_csv(self, filename, **kwargs):
        savetxt(filename, self.geometry[0].detach().cpu().tolist(), **kwargs)

    def save_to_pickle(self, filename, container='array', **kwargs):
        with open(filename, 'wb') as f:
            if container == 'array':
                pickle.dump(self.geometry[0].detach().cpu().tolist(), f)
            elif container == 'dict':
                pickle.dump({'points': self.geometry[0].detach().cpu().tolist()}, f)
            else:
                raise RuntimeError("DeformablePoints.save_to_pickle(): {container} container type not recognized!")
        pass

    def save_to_mesh(self, filename, **kwargs):
        points_count = self.geometry[0].shape[0]
        meshio.write_points_cells(filename, self.geometry[0].detach().cpu().numpy(), [('polygon'+str(points_count), torch.arange(points_count).view(1, -1).numpy())], **kwargs)

    def compute_deformed(self, modules, solver, it, costs=None, intermediates=None):
        assert isinstance(costs, dict) or costs is None
        assert isinstance(intermediates, dict) or intermediates is None

        compound = CompoundModule([self.silent_module, *modules])

        # Compute the deformation cost if needed
        if costs is not None:
            compound.compute_geodesic_control(compound.manifold)
            costs['deformation'] = compound.cost()

        # Shoot the dynamical system
        shoot(Hamiltonian(compound), solver, it, intermediates=intermediates)

        return self._to_deformed(self.module.manifold.gd)

    def _to_deformed(self, gd):
        return (gd,)

    def _to_deformed_new(self, gd):
        return (gd,)


# class DeformablePolylines(DeformablePoints):
#     def __init__(self, points, connections):
#         self.__connections = connections
#         super().__init__(points)

#     @classmethod
#     def load_from_file(cls, filename):
#         pass

#     @property
#     def geometry(self):
#         return (self.module.manifold.gd.detach(), self.__connections)


class DeformableMesh(DeformablePoints):
    def __init__(self, points, triangles):
        self.__triangles = triangles
        super().__init__(points)

    @classmethod
    def load_from_file(cls, filename, dtype=None):
        mesh = meshio.read(filename)
        points = torch.tensor(mesh.points, dtype=dtype)
        triangles = torch.tensor(mesh.cell_dict['triangle'], torch.int)
        return cls(points, triangles)

    def save_to_file(self, filename):
        meshio.write_points_cells(filename, self.silent_module.manifold.gd.detach().numpy(), [('triangle', self.__triangles)])

    @property
    def triangles(self):
        return self.__triangles

    @property
    def geometry(self):
        return (self.silent_module.manifold.gd, self.__triangles)

    def _to_deformed(self, gd):
        return (gd, self.__triangles)

    def _to_deformed_new(self, gd):
        return (gd, self.__triangles)


class DeformableImage(Deformable):
    def __init__(self, bitmap, output='bitmap', extent=None):
        assert isinstance(extent, AABB) or extent is None or isinstance(extent, str)
        assert output == 'bitmap' or output == 'points'

        self.__shape = bitmap.shape
        self.__output = output

        self.__pixel_extent = AABB(0., self.__shape[1]-1, 0., self.__shape[0]-1)

        if extent is None:
            extent = AABB(0., 1., 0., 1.)
        elif isinstance(extent, str) and extent == 'match':
            extent = self.__pixel_extent

        self.__extent = extent

        #pixel_points = pixels2points(self.__extent.fill_count(self.__shape), self.__shape, self.__extent)
        pixel_points = pixels2points(self.__pixel_extent.fill_count(self.__shape), self.__shape, self.__extent)

        self.__bitmap = bitmap
        super().__init__(Landmarks(2, pixel_points.shape[0], gd=pixel_points))

    @classmethod
    def load_from_file(cls, filename, origin='lower', device=None):
        return cls(load_greyscale_image(filename, origin=origin, device=device))

    @classmethod
    def load_from_pickle(cls, filename, origin='lower', device=None):
        pass

    @property
    def geometry(self):
        if self.__output == 'bitmap':
            return (self.bitmap,)
        elif self.__output == 'points':
            return (self.silent_module.manifold.gd, self.__bitmap.flatten()/torch.sum(self.__bitmap))
        else:
            raise ValueError()

    @property
    def shape(self):
        return self.__shape

    @property
    def extent(self):
        return self.__extent
    
    @property
    def pixel_extent(self):
        return self.__pixel_extent

    @property
    def points(self):
        return self.silent_module.manifold.gd

    @property
    def bitmap(self):
        return self.__bitmap

    @property
    def _has_backward(self):
        return True

    def __set_output(self):
        return self.__output

    def __get_output(self, output):
        self.__output = output

    output = property(__set_output, __get_output)

    def _backward_module(self):
        pixel_grid = pixels2points(self.__pixel_extent.fill_count(self.__shape), self.__shape, self.__extent)
        return SilentLandmarks(2, pixel_grid.shape[0], gd=pixel_grid)

    def compute_deformed_new(self, modules, solver, it, costs=None, intermediates=None):
        assert isinstance(costs, dict) or costs is None
        assert isinstance(intermediates, dict) or intermediates is None

        # Forward shooting
        compound_modules = [self.silent_module, *modules]
        compound = CompoundModule(compound_modules)

        shoot(Hamiltonian(compound), solver, it, intermediates=intermediates)
        
        gridpts_defo = self.silent_module.manifold.gd


        if costs is not None:
            costs['deformation'] = compound.cost()

        return self._to_deformed_new(gridpts_defo)

    def _to_deformed_new(self, gd):
        
        gridpts_defo = gd
        gridpts_defo_vec = gridpts_defo.unsqueeze(0).transpose(1,2)
        

        silent_pixel_grid = self._backward_module()
        gridpts = silent_pixel_grid.manifold.gd
        
        normdiff = torch.sum( (gridpts_defo_vec -gridpts.unsqueeze(2))**2, dim=1)
        #ind0 = torch.argmin( normdiff, dim=1)
        _, ind_nearest = torch.topk(normdiff, k=3, dim=1, largest=False)
        
        #project each point on the segment of the two nearest points
        diff_nearest1 = gridpts_defo[ind_nearest[:,1]] - gridpts_defo[ind_nearest[:,0]]
        diff_pts1 = gridpts_defo[ind_nearest[:,0]] - gridpts
        ps1 = torch.sum(diff_nearest1 * diff_pts1, dim=1)
        no1 = torch.sum(diff_nearest1 * diff_nearest1, dim=1)
        t1 = - ps1/no1
        t1 = t1.unsqueeze(1)
        
        
        diff_nearest2 = gridpts_defo[ind_nearest[:,2]] - gridpts_defo[ind_nearest[:,0]]
        diff_pts2 = gridpts_defo[ind_nearest[:,0]] - gridpts
        ps2 = torch.sum(diff_nearest2 * diff_pts2, dim=1)
        no2 = torch.sum(diff_nearest2 * diff_nearest2, dim=1)
        t2 = - ps2/no2
        t2 = t2.unsqueeze(1)

        #proj = gridpts_defo[ind_nearest[:,0]] + 0.5*t1*diff_nearest1 + 0.5*t2*diff_nearest2
        
        #gridpts_defo_inv = torch.mean(gridpts[ind_nearest[:,:]], 1)
        gridpts_defo_inv = gridpts[ind_nearest[:,0]]  + 0.5 * t1 * (gridpts[ind_nearest[:,1]]  - gridpts[ind_nearest[:,0]] )  + 0.5 * t2 * (gridpts[ind_nearest[:,2]]  - gridpts[ind_nearest[:,0]] )
        
        if self.__output == 'bitmap':
            return (deformed_intensities(gridpts_defo_inv, self.__bitmap, self.__extent), )
        elif self.__output == 'points':
            deformed_bitmap = deformed_intensities(gridpts_defo_inv, self.__bitmap, self.__extent)
            return (gd, deformed_bitmap.flatten()/torch.sum(deformed_bitmap))
        else:
            raise ValueError()

    def _to_deformed(self, gd):
        if self.__output == 'bitmap':
            return (deformed_intensities(gd, self.__bitmap, self.__extent), )
        elif self.__output == 'points':
            deformed_bitmap = deformed_intensities(gd, self.__bitmap, self.__extent)
            return (gd, deformed_bitmap.flatten()/torch.sum(deformed_bitmap))
        else:
            raise ValueError()


    def compute_deformed(self, modules, solver, it, costs=None, intermediates=None):
        assert isinstance(costs, dict) or costs is None
        assert isinstance(intermediates, dict) or intermediates is None

        # Forward shooting
        compound_modules = [self.silent_module, *modules]
        compound = CompoundModule(compound_modules)

        shoot(Hamiltonian(compound), solver, it, intermediates=intermediates)

        # Prepare for reverse shooting
        compound.manifold.negate_cotan()

        silent_pixel_grid = self._backward_module()

        # Reverse shooting with the newly constructed pixel grid module
        compound = CompoundModule([silent_pixel_grid, *compound.modules])

        shoot(Hamiltonian(compound), solver, it)

        if costs is not None:
            costs['deformation'] = compound.cost()

        return self._to_deformed(silent_pixel_grid.manifold.gd)


            
            
def deformables_compute_deformed_new(deformables, modules, solver, it, costs=None, intermediates=None):
    assert isinstance(costs, dict) or costs is None
    assert isinstance(intermediates, dict) or intermediates is None

    # Regroup silent modules of each deformable and build a compound module
    silent_modules = [deformable.silent_module for deformable in deformables]
    compound = CompoundModule([*silent_modules, *modules])

    # Forward shooting
    shoot(Hamiltonian(compound), solver, it, intermediates=intermediates)


    # For now, we need to compute the deformation cost after each shooting (and not before any shooting) for computation tree reasons
    if costs is not None:
        costs['deformation'] = compound.cost()

    # Ugly way to compute the list of deformed objects. Not intended to stay!
    deformed = []
    for deformable, silent_module in zip(deformables, silent_modules):
        deformed.append(deformable._to_deformed_new(silent_module.manifold.gd))

    return deformed



def deformables_compute_deformed(deformables, modules, solver, it, costs=None, intermediates=None):
    assert isinstance(costs, dict) or costs is None
    assert isinstance(intermediates, dict) or intermediates is None

    # Regroup silent modules of each deformable and build a compound module
    silent_modules = [deformable.silent_module for deformable in deformables]
    compound = CompoundModule([*silent_modules, *modules])

    # Forward shooting
    shoot(Hamiltonian(compound), solver, it, intermediates=intermediates)

    # Regroup silent modules of each deformable thats need to shoot backward
    # backward_silent_modules = [deformable.silent_module for deformable in deformables if deformable._has_backward]

    shoot_backward = any([deformable._has_backward for deformable in deformables])

    forward_silent_modules = copy.deepcopy(silent_modules)
    # forward_silent_modules = silent_modules

    if shoot_backward:
        # Backward shooting is needed

        # Build/assemble the modules that will be shot backward
        backward_modules = [deformable._backward_module() for deformable in deformables if deformable._has_backward]
        compound = CompoundModule([*silent_modules, *backward_modules, *modules])

        # Reverse the moments for backward shooting
        compound.manifold.negate_cotan()

        # Backward shooting
        shoot(Hamiltonian(compound), solver, it)

    # For now, we need to compute the deformation cost after each shooting (and not before any shooting) for computation tree reasons
    if costs is not None:
        costs['deformation'] = compound.cost()

    # Ugly way to compute the list of deformed objects. Not intended to stay!
    deformed = []
    for deformable, forward_silent_module in zip(deformables, forward_silent_modules):
        if deformable._has_backward:
            deformed.append(deformable._to_deformed(backward_modules.pop(0).manifold.gd))
        else:
            deformed.append(deformable._to_deformed(forward_silent_module.manifold.gd))

    return deformed



def deformables_compute_deformed_multishape(deformables, multishape, constraints, solver, it, costs=None, intermediates=None, labels=None):
    assert isinstance(costs, dict) or costs is None
    assert isinstance(intermediates, dict) or intermediates is None

    # Regroup silent modules of each deformable and build a compound module
    #silent_modules = [[deformable.silent_module for deformable in deformable_list] for deformable_list in deformables]
    #compound = CompoundModule([*silent_modules, *modules])
    
    silent_modules = [multishape.modules[i][0] for i in range(len(deformables))]
    #modules_tot = [[*silent, *mod] for silent, mod in zip(silent_modules, multishape.modules)]
    
    #multi_shape_tot = MultiShape(modules_tot, multishape.sigma_background)
    Ham = MultiShapeHamiltonian.Hamiltonian_multishape(multishape, constraints)
    Ham.geodesic_controls()
    if costs is not None:
        costs['deformation'] = Ham.module.cost()
        
    #print('__________')
    #print(Ham.constraints(Ham.modules.manifold.infinitesimal_action(Ham.modules.field_generator())))
    
    # Forward shooting
    shoot(Ham, solver, it, intermediates=intermediates)


    shoot_backward = any([deformable._has_backward for deformable in deformables])

    #TODO backward
    #forward_silent_modules = copy.deepcopy(silent_modules)
    #print(forward_silent_modules)
    if shoot_backward:
        # Backward shooting is needed
        #TODO: for the moment only one deformable which is an image
        
        # Build/assemble the modules that will be shot backward
        
        # build grid points
        backward_module = deformables[0]._backward_module()
        grid_pts = backward_module.manifold.gd
        
        #labels = model.labels
        grid_pts_deformed = torch.empty([labels.shape[0], backward_module.dim])
        for i, mod in enumerate(multishape.modules[:-1]):
            grid_pts_deformed[labels==i] = mod[0].manifold.gd
        
        i = len(multishape.modules) -1
        
        grid_pts_deformed[labels==i] = multishape.modules[i][-1].manifold.gd  
        gridpts_defo_vec = grid_pts_deformed.unsqueeze(2)
        
        normdiff = torch.sum( (gridpts_defo_vec -grid_pts.unsqueeze(0).transpose(1,2))**2, dim=1)
            
        _, ind_nearest = torch.topk(normdiff, k=1, dim=1, largest=False)
        
        labels_grid = labels[ind_nearest].view(-1)
        
        mod_list = []
        for i, mod in enumerate(multishape.modules[:-1]):
            pt = grid_pts[labels_grid==i].contiguous()
            #mod_list.append(CompoundModule([silent_modules[i], SilentLandmarks(2, pt.shape[0], gd=pt.clone()), *mod.modules[1:]]))
            mod_list.append(CompoundModule([mod.modules[0], SilentLandmarks(2, pt.shape[0], gd=pt.clone()), *mod.modules[1:]]))
        
        i = len(multishape.modules) -1
        pt = grid_pts[labels_grid==i].contiguous()    
        mod_list.append(CompoundModule([*multishape.modules[-1].modules, SilentLandmarks(2, pt.shape[0], gd=pt.clone())]))
        
        # Reverse the moments for backward shooting
        [compound.manifold.negate_cotan() for compound in mod_list]
        multishape_bk = MultiShape.MultiShapeModules(mod_list, multishape.sigma_background, multishape.backgroundtype)
        
        #TODO constraints must be independent from manifold
        Ham = MultiShapeHamiltonian.Hamiltonian_multishape(multishape, constraints)
        # Backward shooting
        shoot(Ham, solver, it)
        
        grid_pts_deformed_bk = torch.empty([labels.shape[0], backward_module.dim])
        for i, mod in enumerate(multishape_bk.modules):
            grid_pts_deformed_bk[labels==i] = mod[1].manifold.gd
          
        
        

    # For now, we need to compute the deformation cost after each shooting (and not before any shooting) for computation tree reasons


    # Ugly way to compute the list of deformed objects. Not intended to stay!
    deformed = []
    #for i in range(len(silent_modules)):
    #for deformable, forward_silent_module in zip(deformables, forward_silent_modules):
    for i, deformable in enumerate(deformables):
        #forward_silent_module = forward_silent_modules[i]
        forward_silent_module = silent_modules[i]
        if deformable._has_backward:
            deformed.append(deformable._to_deformed(grid_pts_deformed_bk))
        else:
            #deformed.append(deformable._to_deformed(deformable.silent_module.manifold.gd))
            deformed.append(deformable._to_deformed(forward_silent_module.manifold.gd))

    return deformed

