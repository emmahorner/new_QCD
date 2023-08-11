#!/usr/bin/env python
# coding: utf-8

import numpy as np
import numba as nb
import os
import matplotlib.pyplot as plt
import spline_gstar
import Num_SH_fast
from Emma3 import sterile_production

#fermion mass
m_e = 0.511
m_mu = 105.7
m_tau = 1777
ve = 0
vmu = 0
vtau = 0
proton = 938.272
neutron = 939.565

#boson mass
pion_pm = 139.57039
pion_0 = 134.9768
kaon_pm = 493.677
kaon_0 = 497.613
eta = 547.862
eta_p = 957.78
rho_pm = 775.11
rho_0 = 775.26

fermions = np.array([[ve, 2], [vmu, 2], [vtau, 2], [m_e, 4], [m_mu, 4], [m_tau, 4], [proton, 4], [neutron, 4] ])
bosons = np.array([[0,2], [pion_pm, 2], [pion_0, 1], [kaon_pm, 2], [kaon_0, 2] , [eta, 1] , [eta_p, 1] , [rho_pm, 6], [rho_0, 3]])


######## g*
@nb.jit(nopython=True)
def g_fermion_integrand(x, m, T): 
    return np.sqrt(x**2+(m/T)**2) * (x**2)/(np.exp(np.sqrt(x**2+(m/T)**2))+1) * np.exp(x)

@nb.jit(nopython=True)
def g_boson_integrand(x, m, T):
    return np.sqrt(x**2+(m/T)**2) * (x**2)/(np.exp(np.sqrt(x**2+(m/T)**2))-1) * np.exp(x)

x_lagauss, w_lagauss = np.polynomial.laguerre.laggauss(40)
@nb.jit(nopython=True)
def rho(m, g, T, integrand):
    rho_vals = (g/2)*T**4/np.pi**2 * integrand(x_lagauss, m, T) * w_lagauss
    return np.sum(rho_vals)

@nb.jit(nopython=True)
def density(T, mass_gf, mass_gb):
    density_val_f = 0 
    density_val_b = 0 
    for massf, gf in mass_gf:
        density_val_f += rho(massf, gf, T, g_fermion_integrand)
    for massb, gb in mass_gb:
        density_val_b += rho(massb, gb, T, g_boson_integrand)
    return density_val_f + density_val_b

@nb.jit(nopython=True)
def gstar(T, mass_gf, mass_gb):
    return 30/np.pi**2*density(T, mass_gf, mass_gb)/T**4



######## g*s
@nb.jit(nopython=True)
def gs_fermion_integrand(x, m, T):
    Ex = np.sqrt(x**2+(m/T)**2)
    return x**2/Ex * (x**2)/(np.exp(Ex)+1) * np.exp(x)

@nb.jit(nopython=True)
def gs_boson_integrand(x, m, T):
    Ex = np.sqrt(x**2+(m/T)**2)
    return x**2/Ex * (x**2)/(np.exp(Ex)-1) * np.exp(x)

x_lagauss, w_lagauss = np.polynomial.laguerre.laggauss(40)
@nb.jit(nopython=True)
def P(m, g, T, integrand): # this is the new rho funciton
    p_vals = g*T**4/(6*np.pi**2) * integrand(x_lagauss, m, T) * w_lagauss
    return np.sum(p_vals)

@nb.jit(nopython=True)
def pressure(T, mass_gf, mass_gb):
    p_vals_f = 0
    p_vals_b = 0
    for massf, gf in mass_gf:
        p_vals_f += P(massf, gf, T, gs_fermion_integrand)
    for massb, gb in mass_gb: 
        p_vals_b += P(massb, gb, T, gs_boson_integrand)
    return p_vals_f + p_vals_b

@nb.jit(nopython=True)
def entropy(T, mass_gf, mass_gb):
    S = (1/T)*(density(T, mass_gf, mass_gb) + pressure(T, mass_gf, mass_gb))
    return S
    
@nb.jit(nopython=True)    
def gstarS(T, mass_gf, mass_gb):
    return 45/(2*np.pi**2)*entropy(T, mass_gf, mass_gb)/T**3
