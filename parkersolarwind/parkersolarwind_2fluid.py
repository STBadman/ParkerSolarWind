import sys
import astropy.units as u
import astropy.constants as const
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numba
import parkersolarwind as psw_funcs

@numba.njit(cache=True)
def parker_polytropic_2fluid(u_,r_, u0_, uc0e_, uc0p_, ug_, gammae_,gammap_, r0_) :
    term1 = 0.5 * (u_**2 - u0_**2)
    term2e = uc0e_**2/(gammae_-1) * (((u0_*r0_**2)/(u_*r_**2))**(gammae_-1) - 1)
    term2p = uc0p_**2/(gammap_-1) * (((u0_*r0_**2)/(u_*r_**2))**(gammap_-1) - 1)
    term3 = -2*ug_**2 * (r0_/r_ - 1)
    return term1 + term2e + term2p + term3

def parker_polytropic_fext_2fluid(u_,r_, ifext_, u0_, uc0e_, uc0p_, ug_, gammae_, gammap_, r0_) :
    ### Warning :
    # Requires inputs to be floats or return floats s.t.:
    # [u_] = km/s, [r_] = R_sun, [ifext_] returns km^2/s^2
    # [u0_] = km/s, [uc0_] = km/s, [ug_] = km/s, [gamma] = None,
    # [r0_] = R_sun
    assert callable(ifext_), "ifext_ must be a two-to-one " \
    "function mapping r1,r2(units=distance) to F.d(units=J/kg)"
    term1 = 0.5 * (u_**2 - u0_**2)
    term2e = uc0e_**2/(gammae_-1) * (((u0_*r0_**2)/(u_*r_**2))**(gammae_-1) - 1)
    term2p = uc0p_**2/(gammap_-1) * (((u0_*r0_**2)/(u_*r_**2))**(gammap_-1) - 1)
    term3 = -2*ug_**2 * (r0_/r_ - 1)
    term4 = -ifext_(r0_,r_)#.to(u.km**2/u.s**2).value
    return term1 + term2e + term2p + term3 + term4

def solve_parker_polytropic_2fluid(
    R_sol, 
    T0e,T0p, 
    gammae,gammap, 
    r0=1.0*u.R_sun, 
    n0=1e7*u.cm**-3, 
    u0=None,
    mu=1
    ) :
    #### NEEDS ADDING : Check solution exists

    ### Note for arbitrary u0 input, the critical point may
    # be undefined

    ### First solve for the critical point
    r_crit = np.nanmax(psw_funcs.solve_sc(T0p, gammap, r0__=r0, mu__=mu))*u.R_sun 
    # Compute sound speed at critical point
    uc_crit = psw_funcs.get_uc_crit(r_crit/r0)

    ### ALLOW ARBITRARY u0 FOR ISOTHERMAL LAYER
    if u0 is None :  u0 = psw_funcs.get_u0(r_crit/r0, gammap, T0p, mu=mu)

    # 1/2 V_esc
    ug = psw_funcs.get_ug(r0=r0)

    # Coronal base sound speed
    uc0e = psw_funcs.get_uc0(gammae, T0e, mu=mu)
    uc0p = psw_funcs.get_uc0(gammap, T0p, mu=mu)
    

    # Solve Bernouilli's equation
    u_sol_polytropic = []
    if np.isnan(uc_crit.value) :
        uguess = [u0.to("km/s").value,2*u0.to("km/s").value]
    else : uguess = [u0.to("km/s").value,uc_crit.to("km/s").value*1.1]

    # Do the unit conversions outside the loop
    args = (u0.to("km/s").value,
            uc0e.to("km/s").value,
            uc0p.to("km/s").value,
            ug.to("km/s").value,
            gammae,gammap,
            r0.to("R_sun").value
           )
    r_crit_val = r_crit.to("R_sun").value
    for R in R_sol.to("R_sun").value :
        sol = opt.root(parker_polytropic_2fluid,
                       uguess,
                       args=(R,) + args
                      )
        if sol.message == 'The solution converged.' :
            uguessnext=sorted(sol.x)
            uguess=uguessnext
        else : uguessnext = [np.nan]*2
        if R < r_crit_val : u_sol_polytropic.append(uguessnext[-2])
        else : u_sol_polytropic.append(uguessnext[-1])
    u_sol_polytropic = np.array(u_sol_polytropic) * u.km/u.s
    u_sol_polytropic[0] = u0
            
    # Produce density and temperature
    rho0 = const.m_p*n0
    rho_sol_polytropic = rho0*(
        u0*r0**2 /(u_sol_polytropic*R_sol**2)
    )
    rho_sol_polytropic[0] = rho0 # In case u
    
    Tp_sol_polytropic = T0p*(
        u0*r0**2/(u_sol_polytropic*R_sol**2)
    )**(gammap-1)
    Tp_sol_polytropic[0] = T0p
    Te_sol_polytropic = T0e*(
        u0*r0**2/(u_sol_polytropic*R_sol**2)
    )**(gammae-1)
    Te_sol_polytropic[0] = T0e           
    
    return (R_sol.to("R_sun"), 
            rho_sol_polytropic.to("kg/cm^3"), 
            u_sol_polytropic.to("km/s"), 
            Te_sol_polytropic.to("MK"),
            Tp_sol_polytropic.to("MK"),
            r_crit.to("R_sun"),
            uc_crit.to("km/s"),
            gammae,
            gammap,
            T0e,
            T0p,
            mu
           ) 


def solve_parker_polytropic_fext_2fluid(
    R_sol, 
    T0e,
    T0p, 
    gammae,
    gammap,
    fext,
    ifext, 
    r0=1.0*u.R_sun, 
    n0=1e7*u.cm**-3, 
    u0=None,
    mu=1
    ) :
    assert callable(fext), "fext must be a one-to-one " \
        "function mapping r(units=distance) to F(units=force/kg)"
    assert callable(ifext), "ifext must be a two-to-one " \
    "function mapping r1,r2(units=distance) to F.d(units=J/kg)"

    ### First solve for the critical point
    r_crit = np.nanmax(psw_funcs.solve_sc_fext(T0p,gammap,
                                     fext,ifext,
                                     r0__=r0,
                                     mu__=mu)
                                     )*u.R_sun 
    # Compute sound speed at critical point
    uc_crit = psw_funcs.get_uc_crit_fext(np.array([r_crit/r0]),fext)[0]

    ### ALLOW ARBITRARY u0 FOR ISOTHERMAL LAYER
    ## If not supplied, then solve for critical point
    if u0 is None :  u0 = psw_funcs.get_u0_fext(np.array([r_crit/r0]), 
                                      gammap, T0p, 
                                      fext, mu=mu)[0]

    # 1/2 V_esc
    ug = psw_funcs.get_ug(r0=r0)

    # Coronal base sound speed
    uc0p = psw_funcs.get_uc0(gammap, T0p, mu=mu)
    uc0e = psw_funcs.get_uc0(gammae, T0e, mu=mu)
    
    # Solve Bernouilli's equation
    u_sol_polytropic = []
    if np.isnan(uc_crit.value) :
        uguess = [u0.to("km/s").value,2*u0.to("km/s").value]
    else : uguess = [u0.to("km/s").value,uc_crit.to("km/s").value*1.1]
    for ii,R in enumerate(R_sol) : 
        sol = opt.root(parker_polytropic_fext_2fluid,
                       uguess,
                       args=(R.to("R_sun").value,
                             ifext,
                             u0.to("km/s").value,
                             uc0e.to("km/s").value,
                             uc0p.to("km/s").value,
                             ug.to("km/s").value,
                             gammae,
                             gammap,
                             r0.to("R_sun").value
                             )
                      )
        if sol.message == 'The solution converged.' :
            uguessnext=sorted(list(sol.x))
            uguess=uguessnext
        else :
            print(R,sol.message) 
            uguessnext = [np.nan]*2
        if R < r_crit : u_sol_polytropic.append(uguessnext[-2])
        else : u_sol_polytropic.append(uguessnext[-1])
    u_sol_polytropic = np.array(u_sol_polytropic) * u.km/u.s
    u_sol_polytropic[0] = u0
            
    # Produce density and temperature
    rho0 = const.m_p*n0
    rho_sol_polytropic = rho0*(
        u0*r0**2 /(u_sol_polytropic*R_sol**2)
    )
    rho_sol_polytropic[0] = rho0 # In case u
    
    Te_sol_polytropic = T0e*(
        u0*r0**2/(u_sol_polytropic*R_sol**2)
    )**(gammae-1)
    Te_sol_polytropic[0] = T0e       

    Tp_sol_polytropic = T0p*(
        u0*r0**2/(u_sol_polytropic*R_sol**2)
    )**(gammap-1)
    Tp_sol_polytropic[0] = T0p   
    
    return (R_sol.to("R_sun"), 
            rho_sol_polytropic.to("kg/cm^3"), 
            u_sol_polytropic.to("km/s"), 
            Te_sol_polytropic.to("MK"),
            Tp_sol_polytropic.to("MK"),
            r_crit.to("R_sun"),
            uc_crit.to("km/s"),
            gammae,
            gammap,
            T0e,
            T0p,
            mu
           ) 

def solve_isothermal_layer_2fluid(R_arr, 
                           R_iso, 
                           Te_iso,
                           Tp_iso, 
                           gammae,
                           gammap, 
                           n0=5e6*u.cm**-3, 
                           mu=1) :
    
    rho0 = mu*const.m_p*n0

    R_iso_ind = np.where(R_arr.to("R_sun").value >= R_iso.to("R_sun").value)[0][0]
    R_arr_iso = R_arr[:R_iso_ind+1]
    R_arr_poly = R_arr[R_iso_ind:]
    
    mu = 1/(1+Te_iso/Tp_iso)

    (_,
    rho_arr_iso, 
    u_arr_iso, 
    _, 
    _) = psw_funcs.solve_parker_isothermal(R_arr_iso,Tp_iso,n0=n0,mu=mu)
    
    Te_arr_iso = Te_iso * np.ones(len(u_arr_iso))
    Tp_arr_iso = Tp_iso * np.ones(len(u_arr_iso)) 

    rho0_poly = rho_arr_iso[-1]
    u0_poly = u_arr_iso[-1]
    Te0_poly = Te_iso
    Tp0_poly = Tp_iso
    gammae=gammae
    gammap=gammap
    r0_poly = R_arr_iso[-1]

    (_,
     rho_arr_poly,
     u_arr_poly,
     Te_arr_poly,
     Tp_arr_poly,
     _,
     _,
     _,
     _,
     _,
     _,
     _
    ) = solve_parker_polytropic_2fluid(
        R_arr_poly,
        Te0_poly,
        Tp0_poly,
        gammae,
        gammap,
        r0_poly,
        n0=rho0_poly/(const.m_p),
        u0=u0_poly,
        mu=1
        )
    
    return (R_arr_iso.to("R_sun"), 
            rho_arr_iso.to("kg/m^3"), 
            u_arr_iso.to("km/s"), 
            Te_arr_iso.to("MK"), 
            Tp_arr_iso.to("MK"),
            R_arr_poly.to("R_sun"), 
            rho_arr_poly.to("kg/m^3"), 
            u_arr_poly.to("km/s"), 
            Te_arr_poly.to("MK"), 
            Tp_arr_poly.to("MK"),
            gammae,gammap, mu)

def solve_isothermal_layer_2fluid_fext(R_arr, 
                                R_iso, 
                                Te_iso,
                                Tp_iso, 
                                gammae,
                                gammap,
                                fext,
                                ifext, 
                                n0=5e6*u.cm**-3, 
                                mu=1,
                                force_free_polytrope=False
                                ) :

    assert callable(fext), "fext must be a one-to-one " \
        "function mapping r(units=distance) to F(units=force/kg)"
    assert callable(ifext), "ifext must be a two-to-one " \
    "function mapping r1,r2(units=distance) to F.d(units=J/kg)"

    rho0 = const.m_p*n0

    R_iso_ind = np.where(R_arr.to("R_sun").value >= R_iso.to("R_sun").value)[0][0]
    R_arr_iso = R_arr[:R_iso_ind+1]
    R_arr_poly = R_arr[R_iso_ind:]
    
    (_,
    rho_arr_iso, 
    u_arr_iso, 
    _, 
    _) = psw_funcs.solve_parker_isothermal_fext(R_arr_iso,Tp_iso,fext,ifext,n0=n0,mu=(1+Te_iso/Tp_iso)**-1)

    Te_arr_iso = Te_iso * np.ones(len(u_arr_iso))
    Tp_arr_iso = Tp_iso * np.ones(len(u_arr_iso)) 

    rho0_poly = rho_arr_iso[-1]
    u0_poly = u_arr_iso[-1]
    T0e_poly = Te_iso
    T0p_poly = Tp_iso
    gammae=gammae
    gammap=gammap
    r0_poly = R_arr_iso[-1]

    if not force_free_polytrope :
        (_,
        rho_arr_poly,
        u_arr_poly,
        Te_arr_poly,
        Tp_arr_poly,
        _,
        _,
        _,
        _,
        _,
        _,
        _
        ) = solve_parker_polytropic_fext_2fluid(
            R_arr_poly,
            T0e_poly,
            T0p_poly,
            gammae,
            gammap,
            fext,
            ifext,
            r0_poly,
            n0=rho0_poly/(const.m_p),
            u0=u0_poly,
            mu=1.0
            )
    else :
        (_,
        rho_arr_poly,
        u_arr_poly,
        Te_arr_poly,
        Tp_arr_poly,
        _,
        _,
        _,
        _,
        _,
        _,
        _
        ) = solve_parker_polytropic_2fluid(
            R_arr_poly,
            T0e_poly,
            T0p_poly,
            gammae,
            gammap,
            r0_poly,
            n0=rho0_poly/(const.m_p),
            u0=u0_poly,
            mu=1.0
            )
    
    return (R_arr_iso.to("R_sun"), 
            rho_arr_iso.to("kg/m^3"), 
            u_arr_iso.to("km/s"), 
            Te_arr_iso.to("MK"),
            Tp_arr_iso.to("MK"), 
            R_arr_poly.to("R_sun"), 
            rho_arr_poly.to("kg/m^3"), 
            u_arr_poly.to("km/s"), 
            Te_arr_poly.to("MK"),
            Tp_arr_poly.to("MK"), 
            gammae,gammap, mu)