import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.interpolate as sp

def save_spline_params(T, g_star, g_star_s, npz_file_name = "Relativistic_Degrees_of_Freedom.npz"):
    if T[0] < T[-1]:
        x = 1/T[::-1]
        g = sp.CubicSpline(x,g_star[::-1])
        gs = sp.CubicSpline(x,g_star_s[::-1])
    else:
        x = 1/T
        g = sp.CubicSpline(x, g_star)
        gs = sp.CubicSpline(x, g_star_s)

    M_spline_gs = np.zeros((len(x)-1,5))
    M_spline_gss = np.zeros((len(x)-1,5))


    for i in range(0,len(x)-1):
        M_spline_gss[i,0] = x[i]
        M_spline_gss[i,1] = gs(x[i])
        M_spline_gss[i,2] = gs(x[i],1)
        M_spline_gss[i,3] = gs(x[i],2)
        M_spline_gss[i,4] = gs(x[i],3)

    for i in range(0,len(x)-1):
        M_spline_gs[i,0] = x[i]
        M_spline_gs[i,1] = g(x[i])
        M_spline_gs[i,2] = g(x[i],1)
        M_spline_gs[i,3] = g(x[i],2)
        M_spline_gs[i,4] = g(x[i],3)

    np.savez(npz_file_name, T = x, g_star = M_spline_gs, g_star_s = M_spline_gss)
    
def spline_funk(x,p):
    dx = (x - p[0]) 
    spline = p[1] + p[2]*dx + (1/2)*p[3]*(dx)**2 + (1/6)*p[4]*(dx)**3
    return spline

def plot_spline_fit(npz_file):
    data = np.load(npz_file)
    x = 1/data['T']
    M_spline_gs = data['g_star']
    M_spline_gss = data['g_star_s']
    
    x_axis = np.zeros(10*(len(x)-2)) 
    y1_axis = np.zeros(10*(len(x) -2))

    y2_axis = np.zeros(10*(len(x) -2))

    for i in range(len(x) - 2):
        xtemp = np.linspace(M_spline_gs[i,0], M_spline_gs[i + 1, 0], 11)
        for j in range (10):
            x_axis[i*10+j] = xtemp[j] 
            y1_axis[i*10+j] = spline_funk(xtemp[j], M_spline_gs[i,:])
            y2_axis[i*10+j] = spline_funk(xtemp[j], M_spline_gss[i,:])
        
    plt.figure()
    plt.loglog(1/x_axis, y1_axis, label=r'$g_*$')
    plt.loglog(1/x_axis, y2_axis, linestyle= '--', label=r'$g_{*s}$')
    plt.xlim(max(1/x_axis), min(1/x_axis))
    plt.xlabel("T (MeV)")
    plt.ylabel(r"$g_*$")
    plt.legend(loc = "upper right")
    