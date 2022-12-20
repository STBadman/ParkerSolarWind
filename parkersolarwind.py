# Packages to be used in this notebook
import sys
import astropy.units as u
import astropy.constants as const
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

##### ISOTHERMAL PARKER SOLAR WIND

def coronal_base_altitude(T_c=5.8e+6*u.K) :
    return (const.G*const.M_sun*const.m_p/(4*const.k_B*T_c)).to("R_sun")

def critical_speed(T_coronal=2e+6*u.K) :
    return ((2*const.k_B*T_coronal/const.m_p)**0.5).to(u.km/u.s)

def critical_radius(T_coronal=2e+6*u.K) :
    return (const.G*const.M_sun/2/critical_speed(T_coronal)**2).to("R_sun")

def parker_isothermal(u,r,u_c,r_c) :
    return ((u/u_c)**2 -1) - np.log((u/u_c)**2) - 4*np.log(r/r_c) -4*(r_c/r-1)

def solve_parker_isothermal(
    R_sol, T0, rho0=1e7*const.m_p/2/u.cm**3, r0=1.0*u.R_sun
    ) :
    u_sol=[]
    R_crit = critical_radius(T0).to("R_sun").value
    u_crit = critical_speed(T0).to("km/s").value

    # There are 4 branches: 
    #      accelerating (u>0, u<0), decelerating (u>0, u<0)
    # Here we look for all 4, throw away the negative ones, 
    # and switch from smaller root to larger root at the critical radius
    u0 = [-u_crit*10,-u_crit/100,u_crit/100,u_crit*10]
    for ii,R in enumerate(R_sol.to("R_sun").value) : 
        u0next=sorted(opt.root(parker_isothermal,u0,
                               args=(R,u_crit,R_crit)).x)
        u0=u0next
        if R < R_crit : u_sol.append(u0next[-2])
        else : u_sol.append(u0next[-1])

    u_sol = np.array(u_sol)*(u.km/u.s)
    rho_sol = rho0 * (u_sol[0]*r0**2)/(u_sol*R_sol.to("R_sun")**2)
    T_sol = T0 * np.ones(len(u_sol))

    return R_sol.to("R_sun"), rho_sol.to("kg/cm^3"), u_sol.to("km/s"), T_sol.to("MK")

###### POLYTROPIC PARKER SOLAR WIND

### define u_g(r0):
def get_ug(r0) : 
    return ((const.G*const.M_sun/(2*r0))**0.5).to("km/s")

### define uc0(gamma,T0)
def get_uc0(gamma,T_0) : 
    return ((gamma*2*const.k_B*T_0/const.m_p)**0.5).to("km/s")

### define f(s_c,T_0, gamma | r0 ) = 0
def s_crit(sc,T0_,gamma_,r0=1*u.R_sun) :
    ug = get_ug(r0).value
    uc0 = get_uc0(gamma_,T0_).value
    term1 = 0.5*((ug/uc0)**(4/(gamma_-1)) * sc**(2*(2*gamma_-3)/(gamma_-1))-1)
    term2 = (1/(gamma_-1))*((uc0/ug)**2*sc - 1)
    term3 = -2*(sc-1)
    return term1 + term2 + term3

### function that solves for s_crit for the accelerating and
### assuming 2 roots and returns [nan,nan] if solution doesn't 
### converge
def solve_sc(T_0__,gamma__,r0__=1.0*u.R_sun) :
    sol = opt.root(s_crit,[1,10],args=(T_0__,gamma__,r0__))
    if sol.message == 'The solution converged.' : return sorted(list(sol.x))
    else : return [np.nan,np.nan]

def get_u0(s_c,gamma,T_0,r0=1*u.R_sun) :
    ug = get_ug(r0)
    uc0 = get_uc0(gamma,T_0)
    b = (3*gamma-5)/(gamma-1)
    return ug * (ug/uc0)**(2/(gamma-1))*np.sqrt(s_c**b)

def get_uc_crit(s_c, r0=1*u.R_sun) : 
    return get_ug(r0)/np.sqrt(s_c)

def parker_polytropic(u_,r_, u0_, uc0_, ug_, gamma_, r0_) :
    term1 = 0.5 * (u_**2 - u0_**2)
    term2 = uc0_**2/(gamma_-1) * (((u0_*r0_**2)/(u_*r_**2))**(gamma_-1) - 1)
    term3 = -2*ug_**2 * (r0_/r_ - 1)
    return term1 + term2 + term3

def solve_parker_polytropic(
    R_sol, 
    T0, 
    gamma, 
    r0=1.0*u.R_sun, 
    rho0=1e7*const.m_p/2/u.cm**3, 
    u0=None
    ) :
    #### NEEDS ADDING : Check solution exists

    ### Note for arbitrary u0 input, the critical point may
    # be undefined

    ### First solve for the critical point
    r_crit = np.nanmax(solve_sc(T0,gamma,r0__=r0))*u.R_sun 
    # Compute sound speed at critical point
    uc_crit = get_uc_crit(r_crit/r0)

    ### ALLOW ARBITRARY u0 FOR ISOTHERMAL LAYER
    if u0 is None :
        ### Otherwise compute the flow speed at r0 
        ### for transonic solution
        u0 = get_u0(r_crit/(1*u.R_sun), gamma, T0)

    # 1/2 V_esc
    ug = get_ug(r0=r0)

    # Coronal base sound speed
    uc0 = get_uc0(gamma, T0)
    
    # Solve Bernouilli's equation
    u_sol_polytropic = []
    uguess = [u0.to("km/s").value,uc_crit.to("km/s").value*1.1]
    for ii,R in enumerate(R_sol) : 
        sol = opt.root(parker_polytropic,
                       uguess,
                       args=(R.to("R_sun").value,
                             u0.to("km/s").value,
                             uc0.to("km/s").value,
                             ug.to("km/s").value,
                             gamma,
                             r0.to("R_sun").value
                             )
                      )
        if sol.message == 'The solution converged.' :
            uguessnext=sorted(list(sol.x))
            uguess=uguessnext
        else : uguessnext = [np.nan]*2
        #print(uguess)
        if R < r_crit : u_sol_polytropic.append(uguessnext[-2])
        else : u_sol_polytropic.append(uguessnext[-1])
    u_sol_polytropic = np.array(u_sol_polytropic) * u.km/u.s
            
    # Produce density and temperature
    rho_sol_polytropic = rho0*(
        u0*r0**2 /(u_sol_polytropic*R_sol**2)
    ) 
    
    T_sol_polytropic = T0*(
        u0*r0**2/(u_sol_polytropic*R_sol**2)
    )**(gamma-1)            
    
    return (R_sol.to("R_sun"), 
            rho_sol_polytropic.to("kg/cm^3"), 
            u_sol_polytropic.to("km/s"), 
            T_sol_polytropic.to("MK"),
            r_crit.to("R_sun"),
            uc_crit.to("km/s"),
           ) 

def solve_isothermal_layer(R_arr, R_iso, T_iso, gamma, rho0=5e6*const.m_p/2/u.cm**3) :
    
    R_arr_iso = R_arr[np.where(R_arr.to("R_sun").value <= R_iso.to("R_sun").value)[0]]
    R_arr_poly = R_arr[np.where(R_arr.to("R_sun").value > R_iso.to("R_sun").value)[0]]
    
    _,rho_arr_iso, u_arr_iso, T_arr_iso = solve_parker_isothermal(R_arr_iso,T_iso)
    
    rho0_poly = rho_arr_iso[-1]
    u0_poly = u_arr_iso[-1]
    T0_poly = T_iso
    gamma=gamma
    r0_poly = R_iso

    (_,
     rho_arr_poly,
     u_arr_poly,
     T_arr_poly,
     _,
     _
    ) = solve_parker_polytropic(
        R_arr_poly,
        T0_poly,
        gamma,
        r0_poly,
        rho0=rho0_poly,
        u0=u0_poly
        )
    
    return (R_arr_iso.to("R_sun"), 
            rho_arr_iso.to("kg/m^3"), 
            u_arr_iso.to("km/s"), 
            T_arr_iso.to("MK"), 
            R_arr_poly.to("R_sun"), 
            rho_arr_poly.to("kg/m^3"), 
            u_arr_poly.to("km/s"), 
            T_arr_poly.to("MK"), 
            gamma)