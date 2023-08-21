import numpy as np
import gstar_gstars as gstar
import matplotlib.pyplot as plt
import scipy.interpolate as sp

T = np.loadtxt("SMgstar.dat", usecols = 0, unpack = True)
gstar_old = np.loadtxt("SMgstar.dat", usecols = 1, unpack = True)
gstars_old = np.loadtxt("SMgstar.dat", usecols = 2, unpack = True)

fit_gstar = sp.CubicSpline(T, gstar_old)
fit_gstars = sp.CubicSpline(T, gstars_old)

def gauss(x, m, s):
    return np.exp(-(x-m)**2/(2*s**2))

def new_gstar(T_QCD, stdT_QCD = 20, stdT_QCD_ratio = 20./180, transition_width_ratio = 2, std_absolute_T = True):
    mu = T_QCD
    if std_absolute_T:
        std = stdT_QCD
    else:
        std = stdT_QCD_ratio * T_QCD
        
    Tup = T_QCD + std * transition_width_ratio
    Tdown = T_QCD - std * transition_width_ratio
    
    highT_index = np.where(T > Tup)
    
    DT_fit = 10
    DT_low = 3
    
    Nhigh = (len(highT_index[0]))
    Nfit = int(np.ceil(np.round(2*std*transition_width_ratio/DT_fit,2)))
    Nlow = int((Tdown-1)//DT_low+1)
    
    
    Tfit = np.zeros(Nhigh + Nfit + Nlow)
    gfit = np.zeros_like(Tfit)
    gsfit = np.zeros_like(Tfit)
    
    for i in range(Nhigh):
        Tfit[i] = T[-(i+1)]
        gfit[i] = gstar_old[-(i+1)]
        gsfit[i] = gstars_old[-(i+1)]
    for i in range(Nfit):
        Tfit[i+Nhigh] = Tup - i * DT_fit
        
    for i in range(Nlow):
        Tfit[i+Nhigh+Nfit] = Tdown - i * DT_low
        gfit[i+Nhigh+Nfit] = gstar.gstar(Tfit[i+Nhigh+Nfit], gstar.fermions, gstar.bosons)
        gsfit[i+Nhigh+Nfit] = gstar.gstarS(Tfit[i+Nhigh+Nfit], gstar.fermions, gstar.bosons)
    
    dgs_qcd = fit_gstar(Tup) - gstar.gstar(Tdown, gstar.fermions, gstar.bosons)
    dgss_qcd = fit_gstars(Tup) - gstar.gstarS(Tdown, gstar.fermions, gstar.bosons)
    
    xx = np.linspace(Tdown, Tup, 10000)
    integral = np.trapz(gauss(xx, mu, std), xx)

    for i in range(Nfit):
        xxx = np.linspace(Tfit[i+Nhigh], Tup, 10000)
        gfit[i+Nhigh] = fit_gstar(Tup) - dgs_qcd/integral * np.trapz(gauss(xxx, mu, std), xxx)
        gsfit[i+Nhigh] = fit_gstars(Tup) - dgss_qcd/integral * np.trapz(gauss(xxx, mu, std), xxx)
    
    return Tfit[::-1], gfit[::-1], gsfit[::-1]

def example_plot():
    Tq = [180*i for i in range(1, 5)]

    Tarr = []
    garr = []
    gsarr = []
    farr = []

    for T_try in Tq:
        Tf, gf, gfs = new_gstar(T_try, std_absolute_T=False)

        Tarr.append(Tf)
        garr.append(gf)
        gsarr.append(gfs)
        farr.append((sp.CubicSpline(Tf, gf), sp.CubicSpline(Tf, gfs)))

    TT = np.logspace(1,np.log10(2000),100000)
    plt.figure()
    for i in range(len(farr)):
        plt.semilogx(TT, farr[i][0](TT), label=r'$T_{QCD} = $'+'{} MeV'.format(Tq[i]))

    plt.xlim(2000,10)
    plt.legend(loc='upper right')
    plt.xlabel("T")
    plt.ylabel(r"$g_*$")
    
    plt.figure()
    for i in range(len(farr)):
        plt.semilogx(TT, farr[i][1](TT), label=r'$T_{QCD} = $'+'{} MeV'.format(Tq[i]))

    plt.xlim(2000,10)
    plt.legend(loc='upper right')
    plt.xlabel("T")
    plt.ylabel(r"$g_{*s}$")