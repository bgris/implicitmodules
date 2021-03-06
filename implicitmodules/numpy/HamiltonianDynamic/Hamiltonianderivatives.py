def dxH(Mod):
    """
    Supposes that Mod is updated, in particluar Mod.Cot and 
    Mod.h (with geodesic control) is filled
    """
    
    v = Mod.field_generator_curr()
    
    ## derivation wrt GD in Xi_GD
    der = Mod.GD.dCotDotV(v)
    
    ## derivation wrt GD in Zeta_GD
    der_fieldgen = Mod.cot_to_innerprod_curr(Mod.GD, 1)
    der.add_cotan(der_fieldgen)
    der.mult_cotan_scal(-1.)
    
    ## derivation of the cost
    der_cost = Mod.DerCost_curr()
    der.add_cotan(der_cost)
    
    return der


def dpH(Mod):
    """
    Supposes that Mod is updated, in particluar Mod.Cot and 
    Mod.h (with geodesic control) is filled
    The derivation wrt p is the application of the field to GD
    """
    
    v = Mod.field_generator_curr()
    appli = Mod.GD.infinitesimal_action(v)
    return appli


def Ham(Mod):
    """
    Supposes that Geodesic controls have been computed and Mod updated
    """
    
    v = Mod.field_generator_curr()
    Mod.Cost_curr()
    return Mod.GD.inner_prod_v(v) - Mod.cost
