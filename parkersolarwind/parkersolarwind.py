# Packages to be used in this notebook
import sys
import astropy.units as u
import astropy.constants as const
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numba

##### ISOTHERMAL PARKER SOLAR WIND

def coronal_base_altitude(T_c=5.8e+6*u.K,mu=0.5) :
    return (const.G*const.M_sun*mu*const.m_p/(2*const.k_B*T_c)).to("R_sun")

def critical_speed(T_coronal=2e+6*u.K,mu=0.5) :
    return ((const.k_B*T_coronal/(mu*const.m_p))**0.5).to(u.km/u.s)

def critical_radius(T_coronal=2e+6*u.K,mu=0.5) :
    return (const.G*const.M_sun/2/critical_speed(T_coronal,mu=mu)**2).to("R_sun")

def critical_radius_fext(f_ext,T_coronal=2e+6*u.K,mu=0.5) :
    assert callable(f_ext), "f_ext must be a one-to-one " \
        "function mapping r(units=distance) to F(units=force/kg)"
    u_c = critical_speed(T_coronal,mu=mu)
    def transcendental(r_c) : 
        r_cu = r_c * u.R_sun
        return ((const.G*const.M_sun/2/r_cu).to("km^2/s^2") 
                - ((r_cu/2)*f_ext(r_cu)).to("km^2/s^2") 
                - u_c.to("km/s")**2
        )
    return opt.root(transcendental,0.5*critical_radius(T_coronal,mu=mu)).x[0]*u.R_sun

@numba.njit(cache=True)
def parker_isothermal(uu,r,u_c,r_c) :
    return ((uu/u_c)**2 -1) - np.log((uu/u_c)**2) - 4*np.log(r/r_c) -4*(r_c/r-1)

def parker_isothermal_fext(uu,r,u_c,r_c,u_g,ifext) :
    ''' not quite as simplified as no closed form expression for r_c'''
    # All args float or function(floats) -> float
    # All speed terms units = km/s
    # All distance terms units = R_sin
    # [ifext(r1,r2)] -> (km/s)^2 

    #u_g = ((const.G*const.M_sun/(r_c*u.R_sun))**0.5).to(u.km/u.s).value 

    assert callable(ifext), "ifext must be a two-to-one " \
        "function mapping r1,r2(units=distance) to F.d(units=J/kg)"

    term1 = 0.5*((uu/u_c)**2-1)
    term2 = -np.log(np.abs(uu*r**2)/(u_c*r_c**2))
    term3 = -u_g**2/u_c**2 * (r_c/r - 1)
    term4 = -ifext(r_c,r)/u_c**2#.to((u.km/u.s)**2).value/u_c**2
    #print(r," : ",term1,term2,term3,term4, " : ", uu)
    return term1 + term2 + term3 +term4 #=0

def solve_parker_isothermal(
    R_sol, T0, n0=1e7*u.cm**-3, r0=1.0*u.R_sun,mu=0.5
    ) :
    u_sol=[]
    R_crit = critical_radius(T0,mu=mu).to("R_sun").value
    u_crit = critical_speed(T0,mu=mu).to("km/s").value
    rho0 = mu*const.m_p*n0 
    ### Note n0 is the total plasma density. Since we are assuming a quasineutral electron-proton
    ### plasma, the proton density and electron densities are both n0/2. This distinction is 
    ### important for energetics.

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

    return R_sol.to("R_sun"), rho_sol.to("kg/cm^3"), u_sol.to("km/s"), T_sol.to("MK"), mu

def solve_parker_isothermal_fext(
    R_sol, T0, fext, ifext, n0=1e7*u.cm**-3, r0=1.0*u.R_sun,mu=0.5
    ) :
    assert callable(fext), "f_ext must be a one-to-one " \
        "function mapping r(units=distance) to F(units=force/kg)"
    assert callable(ifext), "ifext must be a two-to-one " \
        "function mapping r1,r2(units=distance) to F.d(units=J/kg)"
    u_sol=[]
    R_crit = critical_radius_fext(fext, T_coronal=T0, mu=mu).to("R_sun").value
    u_crit = critical_speed(T0, mu=mu).to("km/s").value
    u_g = ((const.G*const.M_sun/(R_crit*u.R_sun))**0.5).to(u.km/u.s).value
    rho0 = mu*const.m_p*n0 
    ### Note n0 is the total plasma density. Since we are assuming a quasineutral electron-proton
    ### plasma, the proton density and electron densities are both n0/2. This distinction is 
    ### important for energetics.

    # There are 4 branches: 
    #      accelerating (u>0, u<0), decelerating (u>0, u<0)
    # Here we look for all 4, throw away the negative ones, 
    # and switch from smaller root to larger root at the critical radius
    u0 = [-u_crit*10,-u_crit/100, u_crit/100, u_crit*10]
    for ii,R in enumerate(R_sol.to("R_sun").value) : 
        u0next=sorted(opt.root(parker_isothermal_fext,u0,
                               args=(R, u_crit, R_crit, u_g, ifext)).x)
        u0=u0next
        if R < R_crit : u_sol.append(u0next[-2])
        else : u_sol.append(u0next[-1])

    u_sol = np.array(u_sol)*(u.km/u.s)
    rho_sol = rho0 * (u_sol[0]*r0**2)/(u_sol*R_sol.to("R_sun")**2)
    T_sol = T0 * np.ones(len(u_sol))

    return R_sol.to("R_sun"), rho_sol.to("kg/cm^3"), u_sol.to("km/s"), T_sol.to("MK"), mu

###### POLYTROPIC PARKER SOLAR WIND

### define u_g(r0):
def get_ug(r0) : 
    return ((const.G*const.M_sun/(2*r0))**0.5).to("km/s")

### define uc0(gamma,T0,mu=0.5)
def get_uc0(gamma,T_0,mu=0.5) : 
    return ((gamma*const.k_B*T_0/(mu*const.m_p))**0.5).to("km/s")

### define f(s_c,T_0, gamma | r0 ) = 0
def s_crit(sc,T0_,gamma_,r0=1*u.R_sun, mu=0.5) :
    ug = get_ug(r0).value
    uc0 = get_uc0(gamma_,T0_,mu=mu).value
    term1 = 0.5*((ug/uc0)**(4/(gamma_-1)) * sc**(2*(2*gamma_-3)/(gamma_-1))-1)
    term2 = (1/(gamma_-1))*((uc0/ug)**2*sc - 1)
    term3 = -2*(sc-1)
    return term1 + term2 + term3

### define f(s_c,T_0, gamma, f_ext(s_c) | r0 ) = 0
def s_crit_fext(sc, T0_, gamma_, fext_, ifext_, r0=1*u.R_sun, mu=0.5) :
    assert callable(fext_), "fext_ must be a one-to-one " \
        "function mapping r(units=distance) to F(units=force/kg)"
    assert callable(ifext_), "ifext_ must be a two-to-one " \
    "function mapping r1,r2(units=distance) to F.d(units=J/kg)"
    ug = get_ug(r0)
    uc0 = get_uc0(gamma_,T0_,mu=mu)
    # Shi+ Eqn 29
    s_prime = (1/sc - ((sc*r0) *fext_((sc*r0).to("R_sun"))
                      ).to(u.km**2/u.s**2).value/(2*ug.to(u.km/u.s).value**2))**-1
    term1 = 0.5*(1/s_prime - (ug/uc0)**(4/(gamma_-1))
                * (sc**4) * (1/s_prime)**((gamma_+1)/(gamma_-1))
                )
    term2 =  1/(gamma_-1)*(1/s_prime - (uc0/ug)**2)
    term3 = -2*(1/sc-1)
    term4 = -1/(ug.to(u.km/u.s)**2).value * ifext_(r0.to("R_sun").value,sc)#*(u.km**2/u.s**2).value
    evaluate = (term1 + term2 + term3 + term4)*-sc
    return evaluate.reshape(sc.shape)

### function that solves for s_crit for the accelerating and
### assuming 2 roots and returns [nan,nan] if solution doesn't 
### converge
def solve_sc(T_0__,gamma__,r0__=1.0*u.R_sun,mu__=0.5) :
    sol = opt.root(s_crit,[1,10],args=(T_0__,gamma__,r0__,mu__))
    if sol.message == 'The solution converged.' : return sorted(list(sol.x))
    else : return [np.nan,np.nan]

def solve_sc_fext(T_0__,gamma__,fext__,ifext__,r0__=1.0*u.R_sun,mu__=0.5) :
    assert callable(fext__), "fext__ must be a one-to-one " \
        "function mapping r(units=distance) to F(units=force/kg)"
    assert callable(ifext__), "ifext__ must be a two-to-one " \
    "function mapping r1,r2(units=distance) to F.d(units=J/kg)"
    
    ### Shi+2022 Eqn 31
    guess = solve_sc(T_0__,gamma__,r0__=r0__,mu__=mu__)
    sol = opt.root(s_crit_fext,guess,args=(T_0__,
                                       gamma__,
                                       fext__,
                                       ifext__,
                                       r0__,mu__))
    if sol.message == 'The solution converged.' : 
        return sorted(list(sol.x))
    else : return [np.nan,np.nan]

def get_u0(s_c,gamma,T_0,r0=1*u.R_sun,mu=0.5) :
    ug = get_ug(r0)
    uc0 = get_uc0(gamma,T_0, mu=mu)
    b = (3*gamma-5)/(gamma-1)
    return ug * (ug/uc0)**(2/(gamma-1))*np.sqrt(s_c**b)

def get_u0_fext(sc,gamma,T0,fext,r0=1*u.R_sun,mu=0.5) :
    assert callable(fext), "fext must be a one-to-one " \
        "function mapping r(units=distance) to F(units=force/kg)"
    # Shi+2022 Equation 30
    ug = get_ug(r0).to("km/s").value
    uc0 = get_uc0(gamma,T0, mu=mu).to("km/s").value
    s_prime = (1/sc - ((sc*r0) *fext((sc*r0).to("R_sun"))
                      ).to(u.km**2/u.s**2).value/(2*ug**2))**-1
    b = (gamma+1)/(gamma-1)
    return ug*(u.km/u.s)  * (ug/uc0)**(2/(gamma-1)) * sc**2 * (1/s_prime)**(b/2)   

def get_uc_crit(s_c, r0=1*u.R_sun) : 
    return get_ug(r0)/np.sqrt(s_c)

def get_uc_crit_fext(sc, fext, r0=1*u.R_sun) : 
    assert callable(fext), "fext must be a one-to-one " \
        "function mapping r(units=distance) to F(units=force/kg)"
    ug = get_ug(r0).to("km/s").value
    s_prime = (1/sc - ((sc*r0) *fext((sc*r0).to("R_sun"))
                      ).to(u.km**2/u.s**2).value/(2*ug**2))**-1
    return get_ug(r0)/np.sqrt(s_prime)

@numba.njit(cache=True)
def parker_polytropic(u_,r_, u0_, uc0_, ug_, gamma_, r0_) :
    term1 = 0.5 * (u_**2 - u0_**2)
    term2 = uc0_**2/(gamma_-1) * (((u0_*r0_**2)/(u_*r_**2))**(gamma_-1) - 1)
    term3 = -2*ug_**2 * (r0_/r_ - 1)
    return term1 + term2 + term3

def parker_polytropic_fext(u_,r_, ifext_, u0_, uc0_, ug_, gamma_, r0_) :
    ### Warning :
    # Requires inputs to be floats or return floats s.t.:
    # [u_] = km/s, [r_] = R_sun, [ifext_] returns km^2/s^2
    # [u0_] = km/s, [uc0_] = km/s, [ug_] = km/s, [gamma] = None,
    # [r0_] = R_sun
    assert callable(ifext_), "ifext_ must be a two-to-one " \
    "function mapping r1,r2(units=distance) to F.d(units=J/kg)"
    term1 = 0.5 * (u_**2 - u0_**2)
    term2 = uc0_**2/(gamma_-1) * (((u0_*r0_**2)/(u_*r_**2))**(gamma_-1) - 1)
    term3 = -2*ug_**2 * (r0_/r_ - 1)
    term4 = -ifext_(r0_,r_)#.to(u.km**2/u.s**2).value
    return term1 + term2 + term3 + term4

def solve_parker_polytropic(
    R_sol, 
    T0, 
    gamma, 
    r0=1.0*u.R_sun, 
    n0=1e7*u.cm**-3, 
    u0=None,
    mu=0.5
    ) :
    #### NEEDS ADDING : Check solution exists

    ### Note for arbitrary u0 input, the critical point may
    # be undefined

    ### First solve for the critical point
    r_crit = np.nanmax(solve_sc(T0,gamma,r0__=r0,mu__=mu))*u.R_sun 
    # Compute sound speed at critical point
    uc_crit = get_uc_crit(r_crit/r0)

    ### ALLOW ARBITRARY u0 FOR ISOTHERMAL LAYER
    if u0 is None :  u0 = get_u0(r_crit/r0, gamma, T0, mu=mu)

    # 1/2 V_esc
    ug = get_ug(r0=r0)

    # Coronal base sound speed
    uc0 = get_uc0(gamma, T0, mu=mu)
    

    # Solve Bernouilli's equation
    u_sol_polytropic = []
    uguess = [u0.to("km/s").value,uc_crit.to("km/s").value*1.1]
    # Do the unit conversions outside the loop
    args = (u0.to("km/s").value,
            uc0.to("km/s").value,
            ug.to("km/s").value,
            gamma,
            r0.to("R_sun").value
           )
    r_crit_val = r_crit.to("R_sun").value
    for R in R_sol.to("R_sun").value :
        sol = opt.root(parker_polytropic,
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
    rho0 = mu*const.m_p*n0
    rho_sol_polytropic = rho0*(
        u0*r0**2 /(u_sol_polytropic*R_sol**2)
    )
    rho_sol_polytropic[0] = rho0 # In case u
    
    T_sol_polytropic = T0*(
        u0*r0**2/(u_sol_polytropic*R_sol**2)
    )**(gamma-1)
    T_sol_polytropic[0] = T0           
    
    return (R_sol.to("R_sun"), 
            rho_sol_polytropic.to("kg/cm^3"), 
            u_sol_polytropic.to("km/s"), 
            T_sol_polytropic.to("MK"),
            r_crit.to("R_sun"),
            uc_crit.to("km/s"),
            gamma,
            T0,
            mu
           ) 


def solve_parker_polytropic_fext(
    R_sol, 
    T0, 
    gamma,
    fext,
    ifext, 
    r0=1.0*u.R_sun, 
    n0=1e7*u.cm**-3, 
    u0=None,
    mu=0.5
    ) :
    assert callable(fext), "fext must be a one-to-one " \
        "function mapping r(units=distance) to F(units=force/kg)"
    assert callable(ifext), "ifext must be a two-to-one " \
    "function mapping r1,r2(units=distance) to F.d(units=J/kg)"

    ### First solve for the critical point
    r_crit = np.nanmax(solve_sc_fext(T0,gamma,
                                     fext,ifext,
                                     r0__=r0,
                                     mu__=mu)
                                     )*u.R_sun 
    # Compute sound speed at critical point
    uc_crit = get_uc_crit_fext(r_crit/r0,fext)

    ### ALLOW ARBITRARY u0 FOR ISOTHERMAL LAYER
    ## If not supplied, then solve for critical point
    if u0 is None :  u0 = get_u0_fext(r_crit/r0, 
                                      gamma, T0, 
                                      fext, mu=mu)

    # 1/2 V_esc
    ug = get_ug(r0=r0)

    # Coronal base sound speed
    uc0 = get_uc0(gamma, T0, mu=mu)
    
    # Solve Bernouilli's equation
    u_sol_polytropic = []
    uguess = [u0.to("km/s").value,uc_crit.to("km/s").value*1.1]
    for ii,R in enumerate(R_sol) : 
        sol = opt.root(parker_polytropic_fext,
                       uguess,
                       args=(R.to("R_sun").value,
                             ifext,
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
        else :
            print(R,sol.message) 
            uguessnext = [np.nan]*2
        if R < r_crit : u_sol_polytropic.append(uguessnext[-2])
        else : u_sol_polytropic.append(uguessnext[-1])
    u_sol_polytropic = np.array(u_sol_polytropic) * u.km/u.s
    u_sol_polytropic[0] = u0
            
    # Produce density and temperature
    rho0 = mu*const.m_p*n0
    rho_sol_polytropic = rho0*(
        u0*r0**2 /(u_sol_polytropic*R_sol**2)
    )
    rho_sol_polytropic[0] = rho0 # In case u
    
    T_sol_polytropic = T0*(
        u0*r0**2/(u_sol_polytropic*R_sol**2)
    )**(gamma-1)
    T_sol_polytropic[0] = T0           
    
    return (R_sol.to("R_sun"), 
            rho_sol_polytropic.to("kg/cm^3"), 
            u_sol_polytropic.to("km/s"), 
            T_sol_polytropic.to("MK"),
            r_crit.to("R_sun"),
            uc_crit.to("km/s"),
            gamma,
            T0,
            mu
           ) 


def solve_isothermal_layer(R_arr, 
                           R_iso, 
                           T_iso, 
                           gamma, 
                           n0=5e6*u.cm**-3, 
                           mu=0.5) :
    
    rho0 = mu*const.m_p*n0

    R_iso_ind = np.where(R_arr.to("R_sun").value >= R_iso.to("R_sun").value)[0][0]
    R_arr_iso = R_arr[:R_iso_ind+1]
    R_arr_poly = R_arr[R_iso_ind:]
    
    (_,
    rho_arr_iso, 
    u_arr_iso, 
    T_arr_iso, 
    _) = solve_parker_isothermal(R_arr_iso,T_iso,n0=n0,mu=mu)
    
    rho0_poly = rho_arr_iso[-1]
    u0_poly = u_arr_iso[-1]
    T0_poly = T_iso
    gamma=gamma
    r0_poly = R_arr_iso[-1]

    (_,
     rho_arr_poly,
     u_arr_poly,
     T_arr_poly,
     _,
     _,
     _,
     _,
     _
    ) = solve_parker_polytropic(
        R_arr_poly,
        T0_poly,
        gamma,
        r0_poly,
        n0=rho0_poly/(mu*const.m_p),
        u0=u0_poly,
        mu=mu
        )
    
    return (R_arr_iso.to("R_sun"), 
            rho_arr_iso.to("kg/m^3"), 
            u_arr_iso.to("km/s"), 
            T_arr_iso.to("MK"), 
            R_arr_poly.to("R_sun"), 
            rho_arr_poly.to("kg/m^3"), 
            u_arr_poly.to("km/s"), 
            T_arr_poly.to("MK"), 
            gamma, mu)


def solve_isothermal_layer_fext(R_arr, 
                                R_iso, 
                                T_iso, 
                                gamma,
                                fext,
                                ifext, 
                                n0=5e6*u.cm**-3, 
                                mu=0.5,
                                force_free_polytrope=False
                                ) :

    assert callable(fext), "fext must be a one-to-one " \
        "function mapping r(units=distance) to F(units=force/kg)"
    assert callable(ifext), "ifext must be a two-to-one " \
    "function mapping r1,r2(units=distance) to F.d(units=J/kg)"

    rho0 = mu*const.m_p*n0

    R_iso_ind = np.where(R_arr.to("R_sun").value >= R_iso.to("R_sun").value)[0][0]
    R_arr_iso = R_arr[:R_iso_ind+1]
    R_arr_poly = R_arr[R_iso_ind:]
    
    (_,
    rho_arr_iso, 
    u_arr_iso, 
    T_arr_iso, 
    _) = solve_parker_isothermal_fext(R_arr_iso,T_iso,fext,ifext,n0=n0,mu=mu)
    
    rho0_poly = rho_arr_iso[-1]
    u0_poly = u_arr_iso[-1]
    T0_poly = T_iso
    gamma=gamma
    r0_poly = R_arr_iso[-1]

    if not force_free_polytrope :
        (_,
        rho_arr_poly,
        u_arr_poly,
        T_arr_poly,
        _,
        _,
        _,
        _,
        _
        ) = solve_parker_polytropic_fext(
            R_arr_poly,
            T0_poly,
            gamma,
            fext,
            ifext,
            r0_poly,
            n0=rho0_poly/(mu*const.m_p),
            u0=u0_poly,
            mu=mu
            )
    else :
        (_,
        rho_arr_poly,
        u_arr_poly,
        T_arr_poly,
        _,
        _,
        _,
        _,
        _
        ) = solve_parker_polytropic(
            R_arr_poly,
            T0_poly,
            gamma,
            r0_poly,
            n0=rho0_poly/(mu*const.m_p),
            u0=u0_poly,
            mu=mu
            )
    
    return (R_arr_iso.to("R_sun"), 
            rho_arr_iso.to("kg/m^3"), 
            u_arr_iso.to("km/s"), 
            T_arr_iso.to("MK"), 
            R_arr_poly.to("R_sun"), 
            rho_arr_poly.to("kg/m^3"), 
            u_arr_poly.to("km/s"), 
            T_arr_poly.to("MK"), 
            gamma, mu)
