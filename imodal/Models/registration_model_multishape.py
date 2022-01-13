from collections import Iterable, OrderedDict

import torch

from imodal.DeformationModules import CompoundModule, Translation, SilentLandmarks
from imodal.Manifolds import CompoundManifold
from imodal.Models import BaseModel, deformables_compute_deformed, deformables_compute_deformed_multishape, DeformableImage
from imodal.MultiShape import MultiShape #import MultiShapeModules
from imodal.MultiShape import MultishapeCompoundManifold


class RegistrationModelMultishape(BaseModel):
    def __init__(self, boundaries, deformables, deformation_modules, attachments, sigma_background, fit_gd=None, lam=1., precompute_callback=None, other_parameters=None, constraints=None, backgroundtype=None):
        
        """
        deformation_modules is a list of N lists of modules, one for each shape
        deformables is a list of lists ????? deformables. The first N correspond to the N shapes, the first element of each list is the boundary
        
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
        self.__backgroundtype = backgroundtype

        if other_parameters is None:
            other_parameters = []


        self.__init_other_parameters = other_parameters
        
        mods = []
        self.__labels = []
        if backgroundtype=='dense':
            # background module is made of dense translations on an area defined by the images in deformable 
            # 
            #build points of local translations
            # TODO: DOES NOT WORK IF OVERLAP 
            # TODO: ONLY one deformable (image) for the moment
            for deformable in deformables:
                #pts = torch.tensor([])
                if isinstance(deformable, DeformableImage):                    
                    #pts = torch.cat([pts,deformable.silent_module.manifold.gd.clone()])
                    pts_translation = deformable.extent.fill_uniform_density(density=1./(0.5*self.__sigma_background)**2)
                    #pts_translation = deformable.silent_module.manifold.gd.clone()
                    pts_grid = deformable.silent_module.manifold.gd.clone()
                    # keep only points inside each boundary
                    labels_grid = torch.zeros(pts_grid.shape[0], dtype=int) + len(boundaries)
                    labels_translation = torch.zeros(pts_translation.shape[0], dtype=int) + len(boundaries)
                    pts_list_grid = []
                    pts_list_translation = []
                    for i, boundary in enumerate(boundaries):
                        lab_grid = boundary.isin_label(pts_grid)
                        labels_grid[lab_grid==True] = i
                        pts_list_grid.append(pts_grid[lab_grid==True].contiguous())
                        lab_translation = boundary.isin_label(pts_translation)
                        labels_translation[lab_translation==True] = i
                        pts_list_translation.append(pts_translation[lab_translation==True].contiguous())
                        #pts_translation = pts_translation[label_translation==False].contiguous()
                    # last one with points outside all boundaries
                    pts_list_grid.append(pts_grid[labels_grid==len(boundaries)]) 
                    pts_list_translation.append(pts_translation[labels_translation==len(boundaries)]) 
                    #pts_list.append(pts_translation) 
            
                    self.__labels = labels_grid
            
            for mod, boundary, pt in zip(deformation_modules, boundaries, pts_list_grid[:-1]):
                #if isinstance(deformable, DeformableImage):
                mods.append(CompoundModule([SilentLandmarks(pt.shape[1], pt.shape[0], gd=pt.clone()), *mod, boundary.silent_module.copy()]))
                #else:
                #    mods.append(CompoundModule([deformable.silent_module.copy(), *mod, boundary.silent_module.copy()]))
            background_pts = torch.cat([boundary.geometry[0].clone() for boundary in boundaries])
            background_pts = torch.cat([background_pts, pts_list_translation[-1].clone()])
            # pts_list_grid[-1].clone()])
                                         
            #background_list = [Translation.Translations(boundary.geometry[0].shape[1], boundary.geometry[0].shape[0], self.__sigma_background, gd=boundary.geometry[0].clone()) for boundary in boundaries]
            #background_list.append(Translation.Translations(pts_list_translation[-1].shape[1], pts_list_translation[-1].shape[0], self.__sigma_background, gd=pts_list_translation[-1].clone()))
            #background_list.append(SilentLandmarks(pts_list_grid[-1].shape[1], pts_list_grid[-1].shape[0], gd=pts_list_grid[-1].clone()))
            
            mods.append(CompoundModule([Translation.Translations(background_pts.shape[1], background_pts.shape[0], self.__sigma_background, gd=background_pts.clone()), SilentLandmarks(pts_list_grid[-1].shape[1], pts_list_grid[-1].shape[0], gd=pts_list_grid[-1].clone()) ]))
            
        elif backgroundtype=='boundary':
            #background module is needed and is made of translations supported by boundaries
            # Maybe check if all constraints are without dense background
            #TODO: only one translation module
            #pts_boundaries = torch.cat([boundary.geometry[0] for boundary in boundaries])
            #background = Translation.Translations(pts_boundaries.shape[1], pts_boundaries.shape[0], self.__sigma_background, gd=pts_boundaries)
            background_pts = torch.cat([boundary.geometry[0].clone() for boundary in boundaries])
            
            #background = CompoundModule([Translation.Translations(boundary.geometry[0].shape[1], boundary.geometry[0].shape[0], self.__sigma_background, gd=boundary.geometry[0].clone()) for boundary in boundaries])
            background = CompoundModule([Translation.Translations(background_pts.shape[1], background_pts.shape[0], self.__sigma_background, gd=background_pts.clone())])
            for deformable, mod, boundary in zip(self.__deformables, deformation_modules, boundaries):
                mods.append(CompoundModule([deformable.silent_module.copy(), *mod, boundary.silent_module.copy()]))
            mods.append(background)
        else:
            # no need for background module
            background = None

            for deformable, mod, boundary in zip(self.__deformables, deformation_modules, boundaries):
                mods.append(CompoundModule([deformable.silent_module.copy(), *mod, boundary.silent_module.copy()]))
        #self.__modules =[*[deformable.silent_module, mod for deformable in self.__deformables], *deformation_modules, background]
        
        

        
        self.__modules = mods #MultiShape.MultiShapeModules(mods, sigma_background)
        self.__init_manifold = MultishapeCompoundManifold.MultishapeCompoundManifold([mod.manifold.clone(False) for mod in self.__modules])
        #self.__init_manifold = MultiShape.MultiShapeModules(self.__modules, sigma_background).manifold
        #[print(manifold) for manifold in self.__init_manifold]
        #print(self.__init_manifold[2].manifolds)
        [[man.cotan_requires_grad_() for  man in manifold] for manifold in self.__init_manifold]
        
        # Update the parameter dict
        self._compute_parameters()

        super().__init__()

    @property
    def modules(self):
        return self.__modules

    @property
    def labels(self):
        return self.__labels

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
    
    def __get_init_manifold(self):
        return self.__init_manifold

    def fill_init_manifold(self, init_manifold):
        self.__init_manifold = init_manifold

    init_manifold = property(__get_init_manifold, fill_init_manifold)

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
            total_cost.backward(retain_graph=True)
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
        
        
        multishape = MultiShape.MultiShapeModules(self.__modules, self.__sigma_background, self.__backgroundtype)
        
        multishape.manifold.fill_gd(self.__init_manifold.gd)
        multishape.manifold.fill_cotan(self.__init_manifold.cotan)
        
        
        return deformables_compute_deformed_multishape(self.__deformables, multishape, self.__constraints, solver, it, costs=costs, intermediates=intermediates, labels=self.__labels)

