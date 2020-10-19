from collections import Iterable, OrderedDict

import torch

from implicitmodules.torch.DeformationModules import CompoundModule
from implicitmodules.torch.Manifolds import CompoundManifold
from implicitmodules.torch.Models import BaseModel, deformables_compute_deformed
from implicitmodules.torch.MultiShape import MultiShape


class RegistrationModelMultishape(BaseModel):
    def __init__(self, deformables, deformation_modules, attachments, sigma_background, fit_gd=None, lam=1., precompute_callback=None, other_parameters=None, constraints=None):
        
        """
        deformation_modules is a list of N lists of modules, one for each shape
        deformables is a list of of lists ????? deformables. The first N correspond to the N shapes, the first element of each list is the boundary
        
        """
        
        #TODO: assert
        #if not isinstance(deformables, Iterable):
        #    deformables = [deformables]

        #if not isinstance(deformation_modules, Iterable):
        #    deformation_modules = [deformation_modules]

        #if not isinstance(attachments, Iterable):
        #    attachments = [attachments]

        #assert len(deformables) == len(attachments)

        self.__deformables = deformables
        self.__attachments = attachments
        self.__deformation_modules = deformation_modules
        self.__precompute_callback = precompute_callback
        self.__fit_gd = fit_gd
        self.__lam = lam
        self.__constraints = constraints
        self.__sigma_background = sigma_background

        if other_parameters is None:
            other_parameters = []

        #TODO build deformable_manifolds
        #deformable_manifolds = [deformable.silent_module.manifold.clone(False) for deformable in self.__deformables]
        #deformation_modules_manifolds = CompoundModule(deformation_modules).manifold.clone(False)

        self.__init_manifold = CompoundManifold([*deformable_manifolds, *deformation_modules_manifolds])

        [manifold.cotan_requires_grad_() for manifold in self.__init_manifold]

        self.__init_other_parameters = other_parameters

        self.__modules = [deformable.silent_module for deformable in self.__deformables]
        self.__modules.extend(deformation_modules)

        # Update the parameter dict
        self._compute_parameters()

        super().__init__()

    @property
    def modules(self):
        return self.__modules

    @property
    def deformation_modules(self):
        return self.__deformation_modules

    @property
    def attachments(self):
        return self.__attachments

    @property
    def precompute_callback(self):
        return self.__precompute_callback

    @property
    def fit_gd(self):
        return self.__fit_gd

    @property
    def init_manifold(self):
        return self.__init_manifold

    @property
    def init_parameters(self):
        return self.__init_parameters

    @property
    def init_other_parameters(self):
        return self.__init_other_parameters

    @property
    def parameters(self):
        return self.__parameters

    @property
    def lam(self):
        return self.__lam

    @property
    def deformables(self):
        return self.__deformables

    def _compute_parameters(self):
        # Fill the parameter dictionary that will be given to the optimizer.

        # For Python version before 3.6, order of dictionary is not garanteed.
        # For Python version 3.6, order is garanteed in the CPython implementation but not standardised in the language
        # For Python beyon version 3.6, order is garanteed by the language specifications
        # Since order for the parameter list is important and to ensure it is preserved with any Python version, we use an OrderedDict
        self.__parameters = OrderedDict()

        # Initial moments
        self.__parameters['cotan'] = {'params': self.__init_manifold.unroll_cotan()}

        # Geometrical descriptors if specified
        if self.__fit_gd:
            self.__parameters['gd'] = {'params': []}

            for fit_gd, init_manifold in zip(self.__fit_gd, self.__init_manifold[len(self.__deformables):]):
                if isinstance(fit_gd, bool) and fit_gd:
                    init_manifold.gd_requires_grad_()
                    self.__parameters['gd']['params'].extend(init_manifold.unroll_gd())
                # Geometrical descriptor is multidimensional
                elif isinstance(fit_gd, Iterable):
                    for fit_gdi, init_manifold_gdi in zip(fit_gd, init_manifold.unroll_gd()):
                        if fit_gdi:
                            self.__parameters['gd']['params'].append(init_manifold_gdi)

        # Other parameters
        self.__parameters.update(self.__init_other_parameters)

    def evaluate(self, target, solver, it, costs=None, backpropagation=True):
        """ Evaluate the model and output its cost.

        Parameters
        ----------
        targets : torch.Tensor or list of torch.Tensor
            Targets we try to approach.
        solver : str
            Solver to use for the shooting.
        it : int
            Number of iterations for the integration.

        Returns
        -------
        dict
            Dictionnary of (string, float) pairs, representing the costs.
        """
        if costs is None:
            costs = {}

        if not isinstance(target, Iterable):
            target = [target]

        assert len(target) == len(self.__deformables)

        # Call precompute callback if available
        precompute_cost = None
        if self.precompute_callback is not None:
            precompute_cost = self.precompute_callback(self.init_manifold, self.modules, self.parameters)

            if precompute_cost is not None:
                costs['precompute'] = precompute_cost

        deformed_sources = self.compute_deformed(solver, it, costs=costs)
        costs['attach'] = self.__lam * self._compute_attachment_cost(deformed_sources, target)

        # if torch.any(torch.isnan(torch.tensor(list(costs.values())))):
        #     print("Registration model has been evaluated to NaN!")
        #     print(costs)

        total_cost = sum(costs.values())

        if total_cost.requires_grad and backpropagation:
            # Compute backward and return costs as a dictionary of floats
            total_cost.backward()
            return dict([(key, costs[key].item()) for key in costs])
        else:
            return costs

    def _compute_attachment_cost(self, deformed_sources, targets, deformation_costs=None):
        return sum([attachment(deformed_source, target.geometry) for attachment, deformed_source, target in zip(self.__attachments, deformed_sources, targets)])

    def compute_deformed(self, solver, it, costs=None, intermediates=None):
        """ Compute the deformed source.

        Parameters
        ----------
        solver : str
            Solver to use for the shooting.
        it : int
            Number of iterations the integration method will do.
        costs : dict, default=None
            If provided, will be filled with the costs associated to the deformation.

        Returns
        -------
        list
            List of deformed sources.
        """
        
        
        multishape = MultiShape(self.__modules, self.__sigma_background)
        
        
        #TODO : change if necessary fill_gd and fill_cotan
        multi_shape.fill_gd(init_manifold.gd)
        multi_shape.fill_cotan(init_manifold.cotan)
        #compound_module = CompoundModule(self.__deformation_modules)
        #compound_module.manifold.fill_gd([manifold.gd for manifold in self.__init_manifold[len(self.__deformables):]])
        #compound_module.manifold.fill_cotan([manifold.cotan for manifold in self.__init_manifold[len(self.__deformables):]])

        #for deformable, deformable_manifold in zip(self.__deformables, self.__init_manifold):
        #    deformable.silent_module.manifold.fill(deformable_manifold)

        return deformables_compute_deformed_multishape(self.__deformables, multishape, self.__constraints, solver, it, costs=costs, intermediates=intermediates)

