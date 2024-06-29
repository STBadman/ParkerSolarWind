import matplotlib.pyplot as plt
from . import parkersolarwind as psw
import astropy.constants as const
import astropy.units as u
import numpy as np

#
def plot_isothermal(iso_sols,fexts=None, lw=2) :

    if fexts is None : fig,axes=plt.subplots(figsize=(12,4),ncols=3,sharex=True)
    else : 
        fig,axes=plt.subplots(figsize=(10,8),ncols=2,nrows=2,sharex=True)
        axes= axes.flatten()

    for ii,sol in enumerate(iso_sols) :
        r_iso, rho_iso, u_iso, T_iso, mu = sol
        T0 = T_iso[0]

        s=axes[1].plot(r_iso,u_iso,linewidth=lw)
        col = s[0].get_color()
        if fexts is None : rc=psw.critical_radius(T0,mu=mu).value
        else : rc = psw.critical_radius_fext(fexts[ii],T_coronal=T0).value
        axes[1].scatter(
            rc, psw.critical_speed(T0,mu=mu).value,
            color= col, s=100, label="$T_0$="+f"{T0}, "+"$\mu=$"+f"{mu}"
            )
        axes[0].plot(r_iso,(rho_iso/(const.m_p/2)).to("cm^-3"),linewidth=lw)
        axes[2].axhline(T0.to("MK").value,color=col,linewidth=lw)

    axes[0].set_ylabel("n (1/cm^3)")
    axes[0].set_title(
        "Density Profile $n_0=$"
        +f"{(rho_iso[0]/(const.m_p/2)).to('cm^-3'):.2E}")
    axes[1].set_ylabel("V$_{SW}$ (km/s)")
    axes[1].set_title("Velocity Profile")
    axes[2].set_ylabel("T (MK)")
    axes[2].set_title("Temperature Profile")

    if fexts is None:
        T_arr = np.logspace(-2,1,40)*u.MK
        axes[1].plot(
            psw.critical_radius(T_arr).value,
            psw.critical_speed(T_arr).value,
            color="black",label="$V_{crit}(T_0)[0.01MK-10MK]$",
            linestyle="--"
            )
        axes[1].legend()

    else : 
        norm0 = const.G*const.M_sun/const.R_sun**2
        for fext in fexts : axes[3].plot(r_iso,fext(r_iso).to(norm0.unit)/norm0)
        axes[3].plot(r_iso,1/r_iso.value**2,color="black",linestyle="--",label="Gravity")
        axes[3].legend()
        axes[3].set_ylabel("F$_{ext}$/(GM$_\odot$/R$_\odot^2$)")
        axes[3].set_title("External Forcing")

    for ax in axes :
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(left=0.9,right=200)
        ax.set_xlabel("R ($R_\odot$)")
        ax.axvline(1,color="black")
        ax.grid(which="both")

    plt.tight_layout()

    return fig,axes

def plot_polytropic(poly_sols,add_iso=None,cm="inferno",fexts=None) :
    
    fig,axes=plt.subplots(figsize=(12,4),ncols=3,sharex=True)

    norm = plt.Normalize(0,len(poly_sols))
    if fexts is None : fexts = [None]*len(poly_sols)
    if add_iso is None : add_iso = [False]*len(poly_sols)
    for jj,(sol,add_iso_,fe) in enumerate(zip(poly_sols,add_iso,fexts)) :
        R_sol,rho_sol,u_sol,T_sol,rcrit,ucrit,gamma,T0,mu = sol
        r0 = R_sol[0]
        if add_iso_ :
            rho_iso, u_iso, T_iso, mu = psw.solve_parker_isothermal(R_sol,T0,mu=mu)[1:]
        if fe is not None : fadd = f"F={fe}"
        else : fadd = ""

        ##### DENSITY #######
        # Polytropic
        axes[0].plot(R_sol,(rho_sol/(const.m_p/2)).to("1/cm^3"),
                     color=plt.get_cmap(cm)(norm(jj))) # 10000/cc at 1Rs
        # Isothermal
        if add_iso_ : 
            s=axes[0].plot(
                R_sol,(rho_iso/(const.m_p/2)).to("1/cm^3"),
                color=plt.get_cmap(cm)(norm(jj)),linestyle="--",)

        ##### VELOCITY ######
        # Isothermal
        if add_iso_ : 
            s=axes[1].plot(
                R_sol,u_iso,
                color=plt.get_cmap(cm)(norm(jj)),linestyle="--"
                )
        # Polytropic
        axes[1].plot(R_sol,u_sol,color=plt.get_cmap(cm)(norm(jj)))
        axes[1].scatter(
            rcrit.to("R_sun"),ucrit.to("km/s"),
            color=plt.get_cmap(cm)(norm(jj)),s=50
            )
        if add_iso_ : 
            axes[1].scatter(
                psw.critical_radius(T0).to("R_sun"),
                psw.critical_speed(T0).to("km/s"),
                color="blue",s=50)    

        ###### TEMPERATURE ######
        # Polytropic
        axes[2].plot(R_sol,T_sol,color=plt.get_cmap(cm)(norm(jj)),
                     label=(f"Polytropic (T0={T0.to('MK'):.1f}, "
                            +"$\gamma$="+f"{gamma}, "+"$\mu$="+f"{mu}), "+fadd)
                    )
        # Isothermal
        if add_iso_ : s=axes[2].plot(
            R_sol,T_iso,
            color=plt.get_cmap(cm)(norm(jj)),linestyle="--",
            label=f"Isothermal (T0={T0:.1f})"
            )

    axes[0].set_ylabel("$n(r)$ (cm$^{-3}$)")
    axes[0].set_title(
        "Density Profile : $n_0$="
        +f"{(rho_sol[0]/(const.m_p/2)).to('cm^-3'):.2E}"
    )

    axes[0].set_ylim(1,2e6)
    
    axes[1].set_ylabel("u(r) (km/s)")
    axes[1].set_title("Velocity Profile")
    axes[1].set_ylim(50,2000)

    axes[2].set_ylabel("T(r) (MK)")
    axes[2].set_title("Temperature Profile")
    axes[2].set_ylim(0.05,10)

    axes[2].legend(ncol=1,fontsize=8)
    for ax in axes :
        ax.set_xlim(right=R_sol[-1].to("R_sun").value,left=0.9)
        ax.grid(which="both")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Heliocentric Distance ($R_\odot$)")
        ax.axvline(1,color="black",linewidth=1.5)
    plt.tight_layout()
    
    return fig,axes

def plot_isothermal_layer(sol,lw=2,figsize=(12,4),fig=None,axes=None,
                          iso_bkg_col = "pink",
                          poly_bkg_col = "cyan",
                          iso_line_col = "red",
                          poly_line_col = "blue",
                          gridlines_opt = "both",
                          force_details=None,
                          add_force_to_legend=True,
                          force_free_crit=False,
                          bkg_alpha=0.3
                          ) :
    
    (R_arr_iso, rho_arr_iso, u_arr_iso, T_arr_iso, 
     R_arr_poly, rho_arr_poly, u_arr_poly, T_arr_poly, gamma, mu) = sol
    R_iso = R_arr_iso[-1]
    T_iso = T_arr_iso[-1]
    
    if force_details is not None :
        mult,fext,ifext = force_details

    if fig is None and axes is None :
        fig,axes=plt.subplots(figsize=figsize,ncols=3,sharex=True)

    #### Proton Number Density (assuming m = m_p/2)
    n_arr_iso = rho_arr_iso/(const.m_p/2)

    axes[0].plot(R_arr_iso.to("R_sun"),
                 n_arr_iso.to("1/cm^3"),
                 color=iso_line_col,linewidth=lw)
    n_arr_poly = rho_arr_poly/(const.m_p/2)
    axes[0].plot(R_arr_poly.to("R_sun"),
                 n_arr_poly.to("1/cm^3"),
                 color=poly_line_col,linewidth=lw)
    
    axes[1].plot(R_arr_iso.to("R_sun"),
                 u_arr_iso.to("km/s"),
                 color=iso_line_col,linewidth=lw,zorder=4)
    if force_details is None :
        axes[1].scatter(psw.critical_radius(T_arr_iso[0],mu=mu).to("R_sun"),
                        psw.critical_speed(T_arr_iso[0],mu=mu).to("km/s"),
                        s=50,color="black",zorder=5)
    else : 
        if add_force_to_legend : 
            if force_free_crit: 
                axes[1].scatter(psw.critical_radius(T_arr_iso[0],mu=mu).to("R_sun"),
                            psw.critical_speed(T_arr_iso[0],mu=mu).to("km/s"),
                            s=50,color="black",zorder=5,label="F=0")
                #lab_force = 
            axes[1].scatter(psw.critical_radius_fext(fext,T_arr_iso[0],mu=mu).to("R_sun"),
                            psw.critical_speed(T_arr_iso[0],mu=mu).to("km/s"),
                            s=50,color="gold",zorder=5,label=f"F={mult}")
            axes[1].legend()
        else:
            if force_free_crit: 
                axes[1].scatter(psw.critical_radius(T_arr_iso[0],mu=mu).to("R_sun"),
                            psw.critical_speed(T_arr_iso[0],mu=mu).to("km/s"),
                            s=50,color="black",zorder=5)
            axes[1].scatter(psw.critical_radius_fext(fext,T_arr_iso[0],mu=mu).to("R_sun"),
                            psw.critical_speed(T_arr_iso[0],mu=mu).to("km/s"),
                            s=50,color="gold",zorder=5)
            axes[1].legend()

    axes[1].plot(R_arr_poly.to("R_sun"),
                 u_arr_poly.to("km/s"),
                 color=poly_line_col,linewidth=lw)
    
    axes[2].plot(R_arr_iso.to("R_sun"),
                 T_arr_iso.to("MK"),
                 color=iso_line_col,linewidth=lw)
    axes[2].plot(R_arr_poly.to("R_sun"),
                 T_arr_poly.to("MK"),
                 color=poly_line_col,linewidth=lw)

    for ax in axes :
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("Radius (Rs)")
        ax.grid(which=gridlines_opt)
        ax.axvline(R_iso.to("R_sun").value,linewidth=2,color="black",linestyle="--")
        ax.axvspan(1,R_iso.to("R_sun").value,color=iso_bkg_col,alpha=bkg_alpha)
        ax.axvspan(R_iso.to("R_sun").value,200,color=poly_bkg_col,alpha=bkg_alpha)
        ax.set_xlim(0.9,200)
        ax.axvline(1,color="black",linewidth=2)

    axes[0].set_ylabel("Number density (cm^-3)")
    axes[1].set_ylabel("Velocity (km/s)")
    axes[2].set_ylabel("Temperature (K)")


    axes[0].set_ylabel("$n(r)$ (cm$^{-3}$)")
    axes[0].set_title(
        "Density Profile : $n_0$="
        +f"{(rho_arr_iso[0]/(const.m_p/2)).to('cm^-3'):.2E}"
    )
    axes[0].set_ylim(1,2e6)

    axes[1].set_ylabel("u(r) (km/s)")
    axes[1].set_title("Velocity Profile")
    axes[1].set_ylim(10,2000)

    axes[2].set_ylabel("T(r) (MK)")
    axes[2].set_title("Temperature Profile")
    axes[2].set_ylim(0.05,10)

    fig.suptitle(
        "Both Layers $R_{iso}$="
        +f"{R_iso:.1f}"+", $T_0=$"+f"{T_iso}"
        +", $\gamma=$"+f"{gamma:.2f}"
        )

    plt.tight_layout()

    return fig, axes
