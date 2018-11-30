# coding: utf-8
from __future__ import division, print_function
import math as mt
import numpy as np
#import scipy.interpolate as spi
import matplotlib.pylab as plt
#import sqrts 
import pickle as pk
import scipy.interpolate as spi
import struct as struct
import sys


############    
# Classes # 
##########
  
# Définition d'une classe f, df pour la recherche de zéros avec algorithme de Newton
class Fdf: # pour les classes, on utilise une majuscule, histoire d'identifier facilement.
    """Définition de la classe fdf : (f,df)"""
    
    #initialisation et définition
    def __init__(self,f=0.,df=0.):
        self.f,self.df = f,df
    
    def variable(x):
        return Fdf(np.array(x),np.ones_like(x))
    variable = staticmethod(variable)

    def constant(x):
        return Fdf(np.array(x),np.zeros_like(x))
    constant = staticmethod(constant)    
    
    def tctfdfc(x):
        """ turn constant to fdf constant. Test if x is a Fdf instance. If not turn to Fdf constant"""
        if isinstance(x,Fdf) :
            pass
        else :  
            x = Fdf.constant(x)
        return x    
    tctfdfc = staticmethod(tctfdfc) 
    
    def reset(self):
        self.f,self.df = 0,0
        return self
    
    def __str__(self):
        return '('+self.f.__str__()+','+self.df.__str__()+')'
        
    # opérations de base (utilisation de +,-,*,/)    
    def __add__(self,a):
        a = Fdf.tctfdfc(a)
        return Fdf(self.f + a.f, self.df+a.df)
    
    def __radd__(self,a):
        a = Fdf.tctfdfc(a)
        return Fdf(self.f + a.f, self.df+a.df)        
    
    def __sub__(self,a):
        a = Fdf.tctfdfc(a)
        return Fdf(self.f-a.f, self.df-a.df)
    
    def __rsub__(self,a):
        a = Fdf.tctfdfc(a)
        return Fdf(self.f-a.f, self.df-a.df)        
    
    def __mul__(self,a):
        a = Fdf.tctfdfc(a)                
        return Fdf(self.f*a.f, self.f*a.df + self.df*a.f)
    
    def __rmul__(self,a):                          
        a = Fdf.tctfdfc(a)
        return Fdf(self.f*a.f, self.f*a.df + self.df*a.f)    
    
    def __div__(self,a):
        a = Fdf.tctfdfc(a)
        return Fdf(self.f/a.f,(self.df*a.f-self.f*a.df)/(a.f*a.f))
    def __rdiv__(self,a):
        a = Fdf.tctfdfc(a)
        return Fdf(a.f/self.f,(self.f*a.df-self.df*a.f)/(self.f*self.f))
    
    def __pow__(self,a):
        return Fdf(self.f**a, a*self.df*self.f**(a-1))     
    
    def __getitem__(self,a) :
        return Fdf(self.f.__getitem__(a),self.df.__getitem__(a))
    def __setitem__(self,a,b) :
        return Fdf(self.f.__setitem__(a,b),self.df.__setitem__(a,b))
    def __getslice__(self,a,b) :
        return Fdf(self.f.__getslice__(a,b),self.df.__getslice__(a,b))
    def __setslice__(self,a,b,c) :
        return Fdf(self.f.__setslice__(a,b,c),self.df.__setslice__(a,b,c))
    
    def scalmul(self,a): # obsolete
        return Fdf(self.f  * a, self.df * a)     
    def sqrt_maystre(v):
        #s = 1j*np.conj(np.sqrt(-np.conj(v.f)))
        s = np.sqrt(1j)*np.sqrt(-1j*np.asarray(v.f,np.complex))
    #    if s.real+s.imag>0:
    #        pass
    #    else:
    #        print 'No Maystre anymore'
    #        print(s)
    #        print(s.real+s.imag)
        # original :
        return Fdf(s,0.5*v.df/s)
    
            
    def square(self): # Pour les scalaires
        return Fdf(self.f**2,2*self.f*self.df)
    
    def squareM(self): # Pour les matrices
        return Fdf(np.mat(self.f,self.f),2*np.multiply(self.f,self.df))
               
    def oppose(self):
        return Fdf(-self.f,-self.df)
        
    def inv(self):
        return Fdf(1/self.f, -self.f**(-2)*self.df)
        
    def cos(self):
        return Fdf(np.cos(self.f),-self.df*np.sin(self.f))
        
    def sin(self):
        return Fdf(np.sin(self), self.df*np.cos(self.f))
        
    def exp(self):
        return Fdf(np.exp(self.f),self.df*np.exp(self.f))
        
    def absolute(self):
        return Fdf(np.abs(self.f),np.real(self.f*np.conj(self.df))/np.abs(self.f))
        
    def squareabs(self):
        return Fdf(self.f*np.conj(self.f),self.f*np.conj(self.df) + np.conj(self.f)*self.df)
    
##############    
# Fonctions # 
############
def sqrt_maystre(v):
    s = np.sqrt(1j)*np.sqrt(-1j*np.asarray(v,np.complex))
    return s
def calc_nb_NP(d,N0):
    """ Calculate the number of something in a nanoparticle\n The shape is assumed to be spherical (diameter d in nm),\n the density of something (N0) is given in cm-3"""
    V_NP = (4*np.pi*((d/2)*1e-7)**3)/3
    return N0*V_NP

def convert_cm_microns(x,y):
    """x, et y vecteurs de même taille. x longueur d'onde, et y données, je fais 1e4/x, puis je reclasse selon les x croissants."""
    R = np.column_stack((x,y))
    R[:,0] = 1e4/R[:,0]
   # R.sort(axis=0) # ca ne marche pas...
    I = np.argsort(R[:,0])
    R=R[I,:]
    return R[:,0],R[:,1]

def convert_Gamma_cm_Tau_s(toto):
    """ Convertit un taux d'amortissement gamma en cm-1 en un temps de collision tau en seconde. gamma prop (1/tau) et vice-versa"""
    return 1./(3e8*toto*1e2)

def convert_eV_cmm3(toto):
    """ assumes epsinf=1 and m*=me """
    hb=1.05458e-34
    ev=1.60218e-19
    eps0    = 8.85e-12
    mel     = 9.109e-31
    return ((toto*ev/(2*np.pi*hb))/ev)**2*eps0*mel/1e6
    
def convert_cmm3_eV(toto):
    """ assumes epsinf=1 and m*=me """
    hb=1.05458e-34
    ev=1.60218e-19
    eps0    = 8.85e-12
    mel     = 9.109e-31
    return (np.sqrt(toto*1e6*ev**2/(eps0*mel)))*2*np.pi*hb/ev

def convert_eV_radps(toto):
    hb=1.05458e-34
    ev=1.60218e-19
    return toto*ev/(2*np.pi*hb)

def convert_eV_s(toto):
    hb=1.05458e-34
    ev=1.60218e-19
    return 1/(toto*ev/hb)

def convert_mev_inv_cm(toto):
    """ convertit de meV en cm-1, et vice-versa """
    hb=1.05458e-34
    ev=1.60218e-19
    c= 3e8
    return toto*ev/(1e5*hb*2*np.pi*c)
        
def convert_mev_microns(toto):
    """ convertit de meV en microns, et vice-versa """
    hb=1.05458e-34
    ev=1.60218e-19
    c= 3e8
    return 1e6*hb*2*np.pi*c/(toto*1e-3*ev)

def convert_s_mev(toto):
    """ convertit de meV en secondes et vice-versa """
    hb=1.05458e-34
    ev=1.60218e-19
    return(2*np.pi*hb*1e3)/(toto*ev)
  
def convert_Tau_mu_GaAs(x):
    """ donne le temps de collision en secondes pour une mobilité donnée en v.cm-2.s-1"""
    m_el = 9.109e-31
    meff = m_el*0.067
    q = 1.6e-19
    x = x*1e-4
    return x*meff/q

def convert_Tau_mu(x,mef):
    """ donne le temps de collision en secondes pour une mobilité donnée en v.cm-2.s-1"""
    m_el = 9.109e-31
    meff = m_el*mef
    q = 1.6e-19
    x = x*1e-4
    return x*meff/q

def convert_mu_Tau_GaAs(x):
    """ donne la mobilité v.cm-2.s-1 pour un gamma en cm-1."""
    m_el = 9.109e-31
    meff = m_el*0.067
    q = 1.6e-19
    mu = q*x/meff
    return mu*1e4
def convert_mu_Tau(mu,mef):
    """ return gamma in cm-1 for a mobility in cm2.V-1.s-1"""
    m_el = 9.109e-31 #kg
    meff = m_el*mef #kg
    q = 1.6e-19 #C
    g = q/(meff*mu) # en s-1 (use E=mc2 : [C][V]=[M][L]2[T]2)
    mu_cm = mu/(3e8*1e2)
    return mu_cm
def convert_conduct_tau(x,N,mef):
    m_el = 9.109e-31
    q = 1.6e-19
    N = N*1e6 #m-3
    return m_el*mef/(N*q**2*x)
    

        
def sqrt_pos(nb) :
    "Racine Carrée avec coupure de Maystre" 
    return sqrt_maystre(nb)
  
    
def fresnel_coef(longueur_onde,kix, epsilon1, epsilon2,polarisation) :
    """ donne les coef r et t de fresnel en fonction de la longueur d'onde et le kix pour la polarisation TM ou TE. Retourne r,t"""  
    ko1 = (2*np.pi/(longueur_onde*1e-9))*(sqrt_maystre(epsilon1))
    ko2 = (2*np.pi/(longueur_onde*1e-9))*(sqrt_maystre(epsilon2))
    k1z = sqrt_pos(ko1**2 - kix**2)
    k2z = sqrt_pos(ko2**2 - kix**2)
    
    if polarisation == 'TE' :
        r = (k1z - k2z)/(k1z + k2z)
        t = (2*k1z)/(k1z + k2z)
    elif polarisation == 'TM' :
        r =  (epsilon2*k1z - epsilon1*k2z)/(epsilon2*k1z + epsilon1*k2z)
        t = 1+r
    return r,t
    

def call_epsilon(longueur_onde, mat, x=0.3,y=0,z=0) :
    """On rentre la longueur d'onde en microns, les materiaux disponibles sont :\n - \n -AlAs\n -SiC\n -InAs \n -InSb \n -InP \n -GaSb \n -AlSb\n -GaAs_ssbandes(x=dopage cm-2,y= w12meV,z= gamma), \n -GaAs_dope(x=dopage en cm-3)\n -AlGaAs_Kim (x = valeurs précises)\n -AlGaAs_Kim_Interp(x = composition Al)\n -Au_Pardo( autour de 3 microns)\n -Au_Etchegoin \n -Au_Ordal \n-Verre (constant à eps = 2.25)"""
    
    eps0=8.854e-12
    mel=9.109e-31
    hb=1.05458e-34
    ev=1.60218e-19
    c= 3e8
    #pi= np.pi de la bibliothèque numpy.
    if  mat == 'GaAs' :
        epsinf = 11
        wL = 292.1
        wT = 268.7
        T= 2.4 #Palik Value
        #T= 1 # Modif
        v=1e4/longueur_onde #cm-1
        epsilon = epsinf*(1+(wL**2-wT**2)/(wT**2-v**2-1j*T*v))
    elif mat =='PbTe':
        epsinf = 32.8
        #eps_s = 388
        wT = 32
        wL = 114.01
        T = 26
        v=1e4/longueur_onde #cm-1
        epsilon = epsinf*(1+(wL**2-wT**2)/(wT**2-v**2-1j*T*v))
    elif  mat == 'GaAs_mod' :
        epsinf = 11
        wL = 291.55
        wT = 268.7
        T= 2.4 #Palik Value
        #T= 1 # Modif
        v=1e4/longueur_onde #cm-1
        epsilon = epsinf*(1+(wL**2-wT**2)/(wT**2-v**2-1j*T*v))
    elif  mat == 'GaAs_Lockwood' :
        epsinf = 10.88
        wL = 292.01
        wT = 268.41
        T1= 2.33 #Palik Value
        T2 = 2.51
        #T= 1 # Modif
        v=1e4/longueur_onde #cm-1
        epsilon = epsinf*((wL**2-v**2+1j*T1*v)/(wT**2-v**2+1j*T2*v))
    elif  mat == 'GaAs_noloss' :
        epsinf = 11
        wL = 292.1
        wT = 267.98
        T= 0 #Palik Value
        #T= 1 # Modif
        v=1e4/longueur_onde #cm-1
        epsilon = epsinf*(1+(wL**2-wT**2)/(wT**2-v**2-1j*T*v))
    elif mat == 'CdO':
        # Rieder et al 1972. 
        wT = 262 #±2
        wL = 478#±25
        T = 1.1 # pas clairement id 
        epsinf = 5.4
        v=1e4/longueur_onde #cm-1   
        epsilon = epsinf*(1+(wL**2-wT**2)/(wT**2-v**2-1j*T*v))
        
    elif mat == 'LiF':
        wT = 305
        wL = np.sqrt((9.02/1.1)*wT**2)
        T = 1.1
        epsinf = 1.93
        v=1e4/longueur_onde #cm-1   
        epsilon = epsinf*(1+(wL**2-wT**2)/(wT**2-v**2-1j*T*v))
    elif mat =='GaP':
        wL = 402.4
        wT = 363.4
        T = 1.1
        epsinf = 9.1
        v=1e4/longueur_onde #cm-1   
        epsilon = epsinf*(1+(wL**2-wT**2)/(wT**2-v**2-1j*T*v))
    elif mat =='AlAs':
        wL = 401.5
        wT = 361.8
        T = 8
        epsinf = 8.2
        v=1e4/longueur_onde #cm-1   
        epsilon = epsinf*(1+(wL**2-wT**2)/(wT**2-v**2-1j*T*v))
    elif mat == 'SiC':
        epsinf  =6.7
        wL      =969 #cm-1
        wT      =793 #cm-1
        T       =4.76 #cm-1
        v=1e4/longueur_onde #cm-1   
        epsilon = epsinf*(1+(wL**2-wT**2)/(wT**2-v**2-1j*T*v))
    elif mat == 'GaN':
        epsinf=5.35
        wL=746
        wT=559
        T=4
        v=1e4/longueur_onde #cm-1   
        epsilon = epsinf*(1+(wL**2-wT**2)/(wT**2-v**2-1j*T*v))
    elif mat == 'InAs':
        epsinf=11.7
        wL=240
        wT=218
        T=4
        v=1e4/longueur_onde #cm-1   
        epsilon = epsinf*(1+(wL**2-wT**2)/(wT**2-v**2-1j*T*v))
    elif mat =='InSb':
        epsinf=15.68
        wL=190.4
        wT=179.1
        T= 2.86
        v=1e4/longueur_onde #cm-1   
        epsilon = epsinf*(1+(wL**2-wT**2)/(wT**2-v**2-1j*T*v))
    elif mat =='InP':
        epsinf=9.61
        wL=345.0
        wT=303.7
        T=3.5
        v=1e4/longueur_onde #cm-1   
        epsilon = epsinf*(1+(wL**2-wT**2)/(wT**2-v**2-1j*T*v))
    elif mat =='InP_Lockwood':
        epsinf=9.61
        wL=345.32
        TL = 0.95
        wT=303.62
        TT=2.80
        v=1e4/longueur_onde #cm-1   
        epsilon = epsinf*((wL**2-v**2-1j*TL*v)/(wT**2-v**2-1j*TT*v))


    elif  mat == 'GaAs_Ideal' :
        epsinf = 10.9
        wL = 291.2
        wT = 267.98
        T= 0 #Palik Value
        #T= 1 # Modif
        v=1e4/longueur_onde #cm-1
        
        epsilon = epsinf*(1+(wL**2-wT**2)/(wT**2-v**2-1j*T*v))
    elif ((mat=='GaSb') | (mat=='AlSb')| (mat=='AlGaSb')):
        if mat=='GaSb':
            x=0
        if mat=='AlSb':
            x=1
        
        v=1e4/longueur_onde #cm-1
        epsinf = 14.5-4.62*x
    # Type GaSb :
        vt1 = 227.1 - 11.7*x -9.6*x**2
        vl1 = 235.7 - 21.6*x - 8.2*x**2
        g1 = 1.01 + 21.78*x -21.43*x**2
        S1 = (vl1**2 -vt1**2)*epsinf

    # Type AlSb 
        vt2 = 312.7 + 14*x - 7.9*x**2
        vl2 = 313+28.6*x -2.1*x**2
        g2 = 8.34 + 2.37*x - 8.84*x**2
        S2 = (vl2**2 -vt2**2)*epsinf
        if mat=='GaSb':
            epsilon = epsinf + S1/(vt1**2 - v**2 - 1j*g1*v)
        if mat=='AlSb':
            epsilon = epsinf + S2/(vt2**2 - v**2 - 1j*g2*v)
        else :
            epsilon = epsinf + S1/(vt1**2 - v**2 - 1j*g1*v) + S2/(vt2**2 - v**2 - 1j*g2*v);
        
    elif mat == 'AlGaAs_Kim_Interp' :
        
        v = 1e4/longueur_onde
        vl1 = 292.7 - 53.61*x+ 21.73*x**2
        vt1 = 268.8 - 10.65*x- 4.746*x**2
        vl2 = 360   + 71.28*x- 29.78*x**2
        vt2 = 358.3 + 7.141*x- 3.649*x**2
        
        gl1 = 3.056 + 8.514*x
        gt1 = 3.761 + 16.62*x #Home made interpolation value
        #gt1 = 3.761 + 2.62*x  # Modif
        gl2 = 12.11 - 8.81*x
        gt2 = 13.7 - 8.817*x #Home made interpolation value
        #gt2 = 10.7 - 8.817*x # Modif
        epsinf = 10.9 -2.544*x + 0.6554*x**2 - 0.8159*x**3
        epsilon = epsinf*(((vl1**2)-(v**2)-1j*v*gl1)*((vl2**2)-(v**2)-1j*v*gl2))/(((vt1**2)-(v**2)-1j*v*gt1)*((vt2**2)-(v**2)-1j*v*gt2))
        
    elif mat == 'AlGaAs_Kim_Interp_noloss' :
        
        v = 1e4/longueur_onde
        vl1 = 292.7 - 53.61*x+ 21.73*x**2
        vt1 = 268.8 - 10.65*x- 4.746*x**2
        vl2 = 360   + 71.28*x- 29.78*x**2
        vt2 = 358.3 + 7.141*x- 3.649*x**2
        
        gl1 = 0
        gt1 = 0 #Home made interpolation value
        #gt1 = 3.761 + 2.62*x  # Modif
        gl2 = 0
        gt2 = 0 #Home made interpolation value
        #gt2 = 10.7 - 8.817*x # Modif
        epsinf = 10.9 -2.544*x + 0.6554*x**2 - 0.8159*x**3
        epsilon = epsinf*(((vl1**2)-(v**2)-1j*v*gl1)*((vl2**2)-(v**2)-1j*v*gl2))/(((vt1**2)-(v**2)-1j*v*gt1)*((vt2**2)-(v**2)-1j*v*gt2))
        
    elif mat == 'AlGaAs_Kim' :
        v=1e4/longueur_onde #cm-1
        if (x==0.14) :
            vt1 = 267.1
            vl1 = 285.7
            gt1 = 5.67
            gl1     = 4.85
            vt2     = 358.8
            vl2     = 369.0
            gt2     = 10.56
            gl2     = 11.31
            epsinf  = 10.57
        
        elif (x==0.18) :
    
            vt1     = 266.9
            vl1     = 283.4
            gt1     = 8.76
            gl1     = 4.24
            vt2     = 360.1
            vl2     = 372.4
            gt2     = 12.20
            gl2     = 10.24
            epsinf  = 10.47
    
        elif(x==0.30) :
    
            vt1     = 265.2
            vl1     = 278.3
            gt1     = 8.64
            gl1     = 6.15
            vt2     = 360.2
            vl2 	= 379.1
            gt2     = 12.10
            gl2     = 9.42
            epsinf  = 10.16
    
        elif(x==0.36) :
    
            vt1     = 264.5
            vl1     = 276.5
            gt1     = 10.69
            gl1     = 5.58
            vt2     = 360.4
            vl2     = 381.3
            gt2     = 12.23
            gl2     = 8.08
            epsinf  = 10.04
    
        elif (x==0.44) :
       
            vt1     = 262.9
            vl1     = 273.7
            gt1     = 10.5
            gl1     = 6.44
            vt2     = 360.2
            vl2     = 385.4 
            gt2     = 9.55 
            gl2     = 7.90
            epsinf  = 9.84
    
        elif(x== 0.54) :
        
            vt1     = 261.8 # Original value
            vl1     = 269.8 # Original value
            gt1     = 12.43 # Original value
            #gt1     = 8
            gl1     = 7.97 # Original value
            #gl1     = 2
            vt2     = 361.5 # Original value
            #vt2     = 357
            vl2     = 390.1 # Original value
            #vl2     = 380
            gt2     = 8.75 # Original value
            gl2     = 8.68 # Original value
            epsinf  = 9.60 # Original value
        else : 
            print ('Pas un bonne valeur de x... seulement dispo : 0.14 0.18 0.3 0.36 0.44 0.54')
            vt1     = 1
            vl1     = 1
            gt1     = 1
            gl1     = 1
            vt2     = 1
            vl2     = 1
            gt2     = 1
            gl2     = 1
            epsinf  = 1
            
        epsilon =  epsinf*(((vl1**2)-(v*v)-1j*v*gl1)*((vl2**2)-(v*v)-1j*v*gl2))/\
        (((vt1**2)-(v*v)-1j*v*gt1)*((vt2**2)-(v*v)-1j*v*gt2))
    
    	
    elif mat == 'Au_Pardo' :
        par = longueur_onde
        lambdap = 1.589540866244842e-07
        gaga = 0.0077
        toto = lambdap/(par*1e-6)
        epsilon = 1 - 1.0/(toto**2 + 1j*gaga)
    elif mat == 'Ag_Nordlander' :
        w = convert_mev_microns(longueur_onde)*1e-3
        sig = 3157.56
        A1 = -1.160e5
        A2 = -4.252
        A3 = -0.4960
        A4 = -2.118
        B1 = -3050
        B2 = -0.8385
        B3 = -13.85
        B4 = -10.23
        C1 = 3.634e8
        C2 = 112.2
        C3 = 1.815
        C4 = 14.31
        #epsilon = 1 + sig/(1j*w) + C1/(w**2 + A1*1j*w + B1) +C2/(w**2 + A2*1j*w + B2)+ C3/(w**2 + A3*1j*w + B3) +C4/(w**2 + A4*1j*w + B4)
        epsilon = 1 - sig/(1j*w) + C1/(w**2 - A1*1j*w + B1) +C2/(w**2 - A2*1j*w + B2)+ C3/(w**2 - A3*1j*w + B3) +C4/(w**2 - A4*1j*w + B4)
    elif mat == 'Au_Nordlander' :
        #
        w = convert_mev_microns(longueur_onde)*1e-3
        sig = 1355.01
        A1 = -8.577e4
        A2 = -2.875
        A3 = -997.6
        A4 = -1.630
        B1 = -1.156e4
        B2 = 0
        B3 = -3090
        B4 = -4.409
        C1 = 5.557e7
        C2 = 2.079e3
        C3 = 6.921e5
        C4 = 26.15
        epsilon = 1 + sig/(1j*w) + C1/(w**2 + A1*1j*w + B1) +C2/(w**2 + A2*1j*w + B2)+ C3/(w**2 + A3*1j*w + B3) +C4/(w**2 + A4*1j*w + B4)
        
    elif mat == 'Au_Etchegoin_mod' :
        #Etchegoin, Le Ru, Meyer, Journal of Chemical Physics125, 164705
        #(2006)
        par = longueur_onde*1e3
        epsinf     = 1.54
        lambdap    = 143      #nm
        gammap     = 14500    #nm
        #A1         = 1.27
        A1         = 1.31
        phi1       = -np.pi/4  #rad
        lambda1    = 470      #nm
        gamma1     = 1900     #nm
        #A2         = 1.1
        A2         = 1
        phi2       = -np.pi/4  #rad
        lambda2    = 325      #nm
        gamma2     = 1060     #nm
        epsilon = epsinf-1/(lambdap**2*((par**(-2))+1j/(gammap*par))) +\
         (A1/lambda1)*( (np.exp(1j*phi1)/(1/lambda1-1/par-1j/gamma1)) +\
         (np.exp(-1j*phi1)/(1/lambda1+1/par+1j/gamma1))) + \
         (A2/lambda2)*( (np.exp(1j*phi2)/(1/lambda2-1./par-1j/gamma2))+ \
         (np.exp(-1j*phi2)/(1/lambda2+1/par+1j/gamma2)))
    elif mat == 'Au_Etchegoin' :
        #Etchegoin, Le Ru, Meyer, Journal of Chemical Physics125, 164705
        #(2006)
        par = longueur_onde*1e3
        epsinf     = 1.54
        lambdap    = 143      #nm
        gammap     = 14500    #nm
        A1         = 1.27
        #A1         = 1.31
        phi1       = -np.pi/4  #rad
        lambda1    = 470      #nm
        gamma1     = 1900     #nm
        A2         = 1.1
        #A2         = 1
        phi2       = -np.pi/4  #rad
        lambda2    = 325      #nm
        gamma2     = 1060     #nm
        epsilon = epsinf-1/(lambdap**2*((par**(-2))+1j/(gammap*par))) +\
         (A1/lambda1)*( (np.exp(1j*phi1)/(1/lambda1-1/par-1j/gamma1)) +\
         (np.exp(-1j*phi1)/(1/lambda1+1/par+1j/gamma1))) + \
         (A2/lambda2)*( (np.exp(1j*phi2)/(1/lambda2-1./par-1j/gamma2))+ \
         (np.exp(-1j*phi2)/(1/lambda2+1/par+1j/gamma2)))    
    elif mat =='GaAs_dope':
        
        par = longueur_onde
        v           = 1e4/par
        epsinf      = 10.9
        wL          = 291.2
        wT          = 267.98
        T           = 2.54
        epsGaAs     =epsinf*(1+(wL**2-wT**2)/(wT**2-v**2-1j*T*v))
        dopage      = x #cm-3
        n3D         = x*1e6 #passage en m-3 %1cm = 1e-2m ; %1cm3 = 1e-6m3, %1cm-3 = 1e6m-3
        meffective  = mel*0.067
        
        # G.Yu et al. ASS 199 p160-165 (2002)
        gammap      = 5.17e-13 + 2.23e-13*np.exp(-dopage/7.62e16) # le dopage est en cm-3 dans cette expression. gaamap est en secondes
        gammap      = y # cm-1 
        
        #gammap      = y
        wpSI        = sqrt_maystre((n3D*ev**2)/(eps0*epsinf*meffective)) #SI : sr.s-1
        wp          = (wpSI*1e-2)/(2*np.pi*c)# conversion cm-1  
            
        eps_elibres = epsinf*wp**2/(v**2 + 1j*v*gammap);
        epsilon         = epsGaAs - eps_elibres;
    elif mat =='Si_dope':
        
        par = longueur_onde
        v           = 1e4/par
        epsinf      = 11.68
        
        dopage      = x #cm-3
        n3D         = x*1e6 #passage en m-3 %1cm = 1e-2m ; %1cm3 = 1e-6m3, %1cm-3 = 1e6m-3
        meffective  = mel*0.29
        
        # G.Yu et al. ASS 199 p160-165 (2002)
        gammap      = 5.17e-13 + 2.23e-13*np.exp(-dopage/7.62e16) # le dopage est en cm-3 dans cette expression. gaamap est en secondes
        gammap      = y # cm-1 
        
        #gammap      = y
        wpSI        = np.sqrt((n3D*ev**2)/(eps0*epsinf*meffective)) #SI : sr.s-1
        wp          = (wpSI*1e-2)/(2*np.pi*c)# conversion cm-1  
            
        eps_elibres = epsinf*wp**2/(v**2 + 1j*v*gammap);
        epsilon         = epsinf - eps_elibres;

       
    elif mat == 'Au_Ordal':
        v = 1e4/longueur_onde
        vp=7.25e4
        g=215
        epsilon = 1-vp*vp/(v*(v+1j*g))
    elif mat == 'Au_Ideal':
        v = 1e4/longueur_onde
        vp=7.25e4
        g=0
        epsilon = 1-vp*vp/(v*(v+1j*g))
    elif mat == 'Verre' :
        epsilon = 2.25*np.ones(np.shape(longueur_onde)) 
    elif mat == 'Verre2' :
        epsilon = 2.0736*np.ones(np.shape(longueur_onde)) 
    elif mat == 'Ti':
        # Ordal AO 24, 4493
        v = 1e4/longueur_onde;
        vp = 2.03e4; #cm-1 ??? moyen ce truc, a vérifier...
        g = 3.82e2; #cm-1
        epsilon = 1-vp*vp/(v*(v+1j*g))
    elif mat =='Ag':
        v = 1e4/longueur_onde
        vp=1e4/(1.24/8.4)
        g=1e-3/(2.99e8*2.2e-14)
        epsilon = 1-vp*vp/(v*(v+1j*g))
    elif mat == 'Al_Ordal':
        v = 1e4/longueur_onde
        vp= 11.9e4
        g= 6.6e2
        epsilon = 1-vp*vp/(v*(v+1j*g))
    elif mat =='Ag_Ordal':
        v       = 1e4/longueur_onde
        vp      = 7.27e4
        g       = 1.45e2
        epsilon = 1-vp*vp/(v*(v+1j*g))
    elif mat =='Ag_Ordal_Ideal':
        v       = 1e4/longueur_onde
        vp      = 7.27e4
        g       = 0
        epsilon = 1-vp*vp/(v*(v+1j*g))
    elif mat =='Cu_Ideal':
        v       = 1e4/longueur_onde
        vp      = 1e-2*11.23e15/(2*np.pi*c)
        g       = 0.732e2
        epsilon = 1-vp*vp/(v*(v+1j*g))
    elif mat =='Al2O3':
        v       = 1e4/longueur_onde
        epsinf  = 3.03
        S1      = 0.42
        O1      = 373.86
        A11     = 24.41
        w011    = 199.33
        G11     = 248.9
        
        S2 = 2.73
        O2 = 439.71
        A12 = 5.68
        w012 = 423.58
        G12 = 38.51
        
        S3 = 2.87
        O3 = 580.42
        A13 = 1.43
        w013 = 371.13
        G13 = 190.22
        A23 = 10.12
        w023 = 536.83
        G23 = 117.73
        A33 = 15.56
        w033 = 914.59
        G33 = 33.25
        A43 = 6.13
        w043 = 1299.6
        G43 = 895.54
        
        S4 = 0.15
        O4 = 638.35
        A14 = 6.77
        w014 = 640.36
        G14 = 30.03
        
        epsilon = epsinf + (S1*O1**2)/(O1**2 - v**2 - 2*O1*Self_energ(A11,w011,G11,v))\
                         + (S2*O2**2)/(O2**2 - v**2 - 2*O2*Self_energ(A12,w012,G12,v))\
                         + (S3*O3**2)/(O3**2 - v**2 - 2*O3*( Self_energ(A13,w013,G13,v)\
                                                           + Self_energ(A23,w023,G23,v)\
                                                           + Self_energ(A33,w033,G33,v)\
                                                           + Self_energ(A43,w043,G43,v)))\
                         + (S4*O4**2)/(O4**2 - v**2 - 2*O4*Self_energ(A14,w014,G14,v))
                                                           
                         
        
    elif ((mat == 'vide') or (mat == 'Vide')) :
        epsilon = np.ones(np.shape(longueur_onde))
    elif mat == 'GaAs_ssbandes' :
        #% retourne la fonction dielectrique de GaAs avec une transition intersousbande en fonction de l'epaisseur de la tranche de gaas et du
        #% dopage de la couche : dopage2D en nb d'e- par cm-2, w12mev en meV, gamma en  s-1
           
        #%% GaAs
        dopage2D = x
        w12meV = y
        gamma = z
        epsinf = 11
        wL = 291.2
        wT = 267.98
        T= 2.4
        v=1e4/longueur_onde #cm-1
       
        epsGaAs = epsinf*(1+(wL**2-wT**2)/(wT**2-v**2-1j*T*v))
        
        if dopage2D == 0:
            eps_ssbandes = 0
        elif w12meV ==0:
            eps_ssbandes =0
        else :
    
            meffective          = 0.0636*mel #kg
            d_eq                = 3.8e-9 # m
            force_oscillateur   = (2*meffective/hb)*((w12meV*1e-3*ev/hb)*d_eq**2) # SI
            ns                  = (dopage2D/20e-7)*1e6 # passage de cm-2 ? m-2 On fixe l'epaisseur du puit a 20nm, ordre de grandeur...
            wp2                 = (ns*force_oscillateur*ev**2)/(meffective*eps0) # SI
            w                   = 2*np.pi*c/(longueur_onde*1e-6) # SI
            eps_ssbandes        = wp2/(w**2 - (w12meV*1e-3*ev/hb)**2 + 1j*gamma*w)
            
        epsilon             = epsGaAs - eps_ssbandes    

    else:
        print ('A pas compris le materiau ton eps est = 0')
        epsilon = 0
    
#    for index in range(len(epsilon)) :
#        
#        if type(epsilon[index]) == np.complex128 :
#            if epsilon[index].imag<0 :
#                epsilon[index] = epsilon[index].conjugate
                
    if type(epsilon)==complex:
        epsilon = epsilon.real + 1j*np.abs(epsilon.imag)            
    return epsilon
    

   

        
    
def create_matvar(longueur_onde, epsilon,Nom_fic) :
    """ Create a file containing wavelength (microns) real part of n and imaginary part of n.test pour voir si long_onde est un array ou un simple integer.  De préférence Nom_fic doit être 'matvar1.dat' ou 'matvar2.dat'"""

    f = open(Nom_fic,'w')
    #nk = sqrts.posimag(epsilon) # en prenant posimag on s'assure une partie imaginaire positive
    nk = np.sqrt(epsilon)
    nk.real = np.abs(nk.real)
    nk.imag = np.abs(nk.imag)
    compt=0        
    while (compt<=len(longueur_onde)-1) :
        f.write(str(longueur_onde[compt]))
        f.write('    ')
        f.write( str(nk.real[compt]))
        f.write('    ')
        f.write( str(nk.imag[compt]))
        f.write('\n') 
        compt = compt+1
    f.close()
    return
    
def print_model(model) :

    fid = open('./simon/boucle1.don','w')
    fid.write('RESPECTEZ ABSOLUMENT                     \n')          
    fid.write('LA PRESENTATION                          \n')              
    fid.write('Nb ordres : impaire N =                  \n')       
    fid.write('%.0f                   \n' %(np.ceil(model['ordre'])))
    fid.write('Nb de couches M =                        \n')             
    fid.write(str(model['nbcouche']) + '                \n')
    fid.write('Discretization parameter s (Nb=s*N)      \n')
    fid.write(str(model['discparam']) +'                \n')
    fid.write('type primary boucle(wavelength)          \n')
    fid.write("'" + str(model['typeboucle1']) + "'" + ' \n')
    fid.write('Debut de boucle                          \n')
    fid.write(str(model['debutboucle1']) +'D-6          \n')
    fid.write('FIN de boucle                            \n')
    fid.write(str(model['finboucle1']) + 'D-6           \n')
    fid.write('Pas de boucle                            \n')
    fid.write(str(model['pasboucle1']) + 'D-6           \n')
    fid.write('Auxiliary val                            \n')
    fid.write('type secondary boucle(angle_theta)       \n')
    fid.write( "'" + str(model['typeboucle2']) + "'" + '\n')
    fid.write('Debut de boucle                          \n')
    fid.write(str(model['debutboucle2']) +'D0          \n')
    fid.write('FIN de boucle                            \n')
    fid.write(str(model['finboucle2']) +'D0            \n')
    fid.write('Pas de boucle                            \n')
    fid.write(str(model['pasboucle2']) +'D0            \n')
    fid.write('Auxiliary val                            \n')
    fid.write( ' 1                                      \n')
    fid.write( ' 1                                      \n')
    fid.close()    

def print_reseau(rescst,reseau,model) :
    """ imprime le fichier reseau.don. lambda_res et lambda_onde en microns, les angles sont en D0, epaisseur des couche en nm """
    fid = open('./simon/reseau1.don','w')
    fid.write('RESPECTEZ ABSOLUMENT                           \n')         
    fid.write('LA PRESENTATION                                \n')
    fid.write('Lambda_reseau =                                \n')
    fid.write(str(rescst['lambda_res'])+ 'D-6                 \n')
    fid.write('Lambda_onde =                                  \n')
    fid.write(str(rescst['lambda_onde'])+ 'D-6                \n')
    fid.write('angles_theta                                   \n')
    fid.write('(en degres)                                    \n')
    fid.write(str(rescst['theta'])+ 'D0                       \n')
    fid.write('angles_phi                                     \n')
    fid.write('(en degres)                                    \n')
    fid.write(str(rescst['phi']) +'D0                         \n')
    fid.write('angles_psi                                     \n')
    fid.write('(en degres)                                    \n')
    fid.write(str(rescst['psi']) +'D0                         \n')
    fid.write('material pour z<Zmin                           \n')
    fid.write('name (in case we interpolate)                  \n')
    fid.write(str(rescst['matZmin'])+ '	                      \n')
    fid.write('reflect index (in case we do not)              \n')
    fid.write('(1.0D0,0.0D0)                                  \n')
    fid.write('interpolation  (no,normal)                     \n')
    fid.write( "'no'" +'                                      \n')
    fid.write('indice pour z>Zmax                             \n')
    fid.write('name (in case we interpolate)                  \n')
    fid.write(str(rescst['matZmax'])+ '                       \n')
    fid.write('reflect index (in case we do not)              \n')
    fid.write('(3.20D0,0.100000)                              \n')
    fid.write('interpolation (no,normal)                      \n')
    fid.write("'normal'"+'                                    \n')
    fid.write('Type layer input                               \n')
    fid.write("'generallayers'" + '                           \n')
    
    ncche = 0            
    while ncche <= model['nbcouche']-1 :
        fid.write('Next Layer                                 \n')
        fid.write( "'simplelayer'"+ '                         \n')
        fid.write('LAYER3                                     \n')
        fid.write('thickness                                  \n')
        fid.write(str(reseau['couche_epaisseur'][ncche])+'D-9 \n')
        fid.write('number of materials                        \n')
        fid.write(str(reseau['nbmat'][ncche])+ '              \n')
        nummat = 0
        while nummat <= (reseau['nbmat'][ncche]-1) :
            fid.write('First Material                         \n')
            fid.write('name (in case we interpolate)          \n')
            fid.write(str(reseau['mat'][ncche][nummat])+ '    \n')
            fid.write('reflective constant (in case we do not)\n')
            fid.write('(1.0D0,0.00D0)                         \n')
            fid.write('interpolation  (no,normal)             \n')
            if (str(reseau['mat'][ncche][nummat]) =='vide') :
                fid.write("'no'"+ '                     \n')
            else :
                fid.write("'normal'"+ '                 \n')
            
            fid.write('End of this point(last layer : one)     \n')
            fid.write(str(reseau['endpoints'][ncche][nummat])+'\n')
            nummat = nummat+1
        ncche = ncche+1              
    fid.close ()   

def print_result_simple(rescst) :
#epaisseur_autour en microns
    fid = open('./simon/results1.don','w')
    fid.write('RESPECTEZ ABSOLUMENT                                                \n')         
    fid.write('LA PRESENTATION                                                     \n')
    fid.write('Save all the layer structure each time? (yes or no)                 \n')
    fid.write(  " 'no' " +'                                                        \n')
    fid.write('Write the changes (Er..) for each layer of the structure (yes or no)\n')
    fid.write(  " 'no' " +'                                                        \n')
    fid.write('What to print (all,simple...)                                       \n')
    fid.write(  " 'simple' "+ '                                                    \n')
    fid.write('How to calculated absorption (none if not)                          \n')
    fid.write(  " 'none' " +'                                                      \n')
    fid.write('What absorption to calculate(All,Total)                             \n')
    fid.write(  " 'none' " +'                                                      \n')
    fid.write('How to plot (none if not, normal)                                   \n')
    fid.write(  " 'none' " +'                                                      \n')
    fid.write('What to plot (All)                                                  \n')
    fid.write(  " 'none' " +'                                                      \n')
    fid.write('Thickness to plot before and after the structure                    \n')
    fid.write( str(rescst['epaisseur_autour']) +'D-6                               \n')
    fid.write('How to discretize the layers                                        \n')
    fid.write(  " 'fixednbpoints' " '                                              \n')
    fid.write('Value quantizazing the former (nb of points, distance)              \n')
    fid.write(  str(rescst['Zdiscretization']) +'                                  \n')
    fid.write( "Select order  'none','allamplitudes','oneorderamplitude','onelayeramplitude','oneorderabsorption','onelayerabsorption','allabsorptions', 'onelayerabsorptionamplitude'\n")
    fid.write("'none'\n")
    fid.write('0')
    fid.close ()
    
def print_result_champ(rescst) :
#epaisseur_autour en microns
    fid = open('./simon/results1.don','w')
    fid.write('RESPECTEZ ABSOLUMENT                                                \n')         
    fid.write('LA PRESENTATION                                                     \n')
    fid.write('Save all the layer structure each time? (yes or no)                 \n')
    fid.write(  " 'no' " +'                                                        \n')
    fid.write('Write the changes (Er..) for each layer of the structure (yes or no)\n')
    fid.write(  " 'no' " +'                                                        \n')
    fid.write('What to print (all,simple...)                                       \n')
    fid.write(  " 'all' "+ '                                                       \n')
    fid.write('How to calculated absorption (none if not)                          \n')
    fid.write(  " 'none' " +'                                                      \n')
    fid.write('What absorption to calculate(All,Total)                             \n')
    fid.write(  " 'none' " +'                                                      \n')
    fid.write('How to plot (none if not, normal)                                   \n')
    fid.write(  " 'normal' " +'                                                    \n')
    fid.write('What to plot (All)                                                  \n')
    fid.write(  " 'All' " +'                                                       \n')
    fid.write('Thickness to plot before and after the structure                    \n')
    fid.write( str(rescst['epaisseur_autour']) +'D-6                               \n')
    fid.write('How to discretize the layers                                        \n')
    fid.write(  " 'fixednbpoints' " '                                              \n')
    fid.write('Value quantizazing the former (nb of points, distance)              \n')
    fid.write(  str(rescst['Zdiscretization']) +'                                  \n')
    fid.close ()
    
def print_result_abs(rescst) :
#epaisseur_autour en microns
    fid = open('./simon/results1.don','w')
    fid.write('RESPECTEZ ABSOLUMENT                                                \n')         
    fid.write('LA PRESENTATION                                                     \n')
    fid.write('Save all the layer structure each time? (yes or no)                 \n')
    fid.write(  " 'no' " +'                                                        \n')
    fid.write('Write the changes (Er..) for each layer of the structure (yes or no)\n')
    fid.write(  " 'no' " +'                                                        \n')
    fid.write('What to print (all,simple...)                                       \n')
    fid.write(  " 'simple' "+ '                                                       \n')
    fid.write('How to calculated absorption (none if not)                          \n')
    fid.write(  " 'integrateanalytical' " +'                                                      \n')
    fid.write('What absorption to calculate(All,Total)                             \n')
    fid.write(  " 'All' " +'                                                      \n')
    fid.write('How to plot (none if not, normal)                                   \n')
    fid.write(  " 'normal' " +'                                                    \n')
    fid.write('What to plot (All)                                                  \n')
    fid.write(  " 'All' " +'                                                       \n')
    fid.write('Thickness to plot before and after the structure                    \n')
    fid.write( str(rescst['epaisseur_autour']) +'D-6                               \n')
    fid.write('How to discretize the layers                                        \n')
    fid.write(  " 'fixednbpoints' " '                                              \n')
    fid.write('Value quantizazing the former (nb of points, distance)              \n')
    fid.write(  str(rescst['Zdiscretization']) +'                                  \n')
    fid.close ()
 
def smooth_curve(x,y) :
    """ smooth la courbe. Chaque point est la moyenne des 4 voisins autour. (sauf aux extremités)"""
    yy = y.copy()
    for i in range(len(y)-2) :
        if i==0 :
            yy[0] = y[0]
        elif i==1 :
            yy[1] = (y[0] + y[1] + y[2])/3
        else :
            yy[i] = (y[i-2]+y[i-1]+y[i]+y[i+1]+y[i+2])/5
    
    yy[len(y)-2] = y[-2]
    yy[len(y)-1] = y[-1]
    return yy

def smooth_curve_n(y,nb=5,it=1) :
    """ smooth_curve(y,nb=5,it=1), smooth la courbe. Chaque point est la moyenne des 4 voisins autour. (sauf aux extremités). it est le nombre d'itération souhaitée, nb est le nombre de points voisins, nb doit etre inf a len(y)*2"""
    def smooth_one(y):
        yy = y.copy()
        # for simplicity, lets just don't do anything to the first and last points.
        for i in range(len(y)-nb):
            if i<nb :
                pass
            else :
                #indexes = np.linspace(i-nb,i+nb,2*nb+1)
                #yy[i] = np.mean(yy[i-nb:i+nb])
                yy[i] = np.mean(yy[i-nb:i+nb+1])  
        return yy
     
    for ii in range(it):
        y = smooth_one(y)
     
    return y
     
def Read_RealTime_JPK(Fullpath):
    """ Return a a dictionnary file with keys."""
    fs = open(Fullpath,'r')
    txt = fs.readline()
    #freq = np.float(txt[18:]) #sample rate  in Hz
    
    txt = fs.readline()
    #txt = fs.readline()
    ii = -1
    indexes = []
    for truc in txt:
        ii = ii+1
        if truc ==' ':
            indexes.append(ii)
    
    columns_name = []
    ii = -1
    for index in indexes:
        ii = ii+1
        if ii >=2: #skip first space that does not count and start at the second
            # modify the names
            col_name = txt[indexes[ii-1]:index].replace('"','')
            col_name = col_name.replace(' ','')
            columns_name.append(col_name)
    # miss the last
    columns_name.append(txt[index:].replace('\r\n',''))
    
    Mat = {}
    for name in columns_name:
            Mat[name] = []
            
            
    while 1: 
        txt = fs.readline()
        if (txt =='#\r\n'): 
                break
    # Data begins
    
    
    while 1: 
        txt = fs.readline()
        #print(txt)
        if ((txt =='')|(txt == '\r\n')): 
            break
        #print(txt)
        ii = -1
        indexes = []
        indexes.append(0) # for the rest need a starting point
        for truc in txt:
            ii = ii+1
            if truc ==' ':
                indexes.append(ii)
        indexes.append(len(txt)) # to get the last
        ii = 0 
        for name in columns_name:
            Mat[name].append(np.float(txt[indexes[ii]:indexes[ii+1]]))
            ii = ii+1
    fs.close()
    return Mat    
    
def Read_Spectrum(Path,borne1 = 0,borne2 = 0) :
    """ Charge un fichier provenant d'un fichier exporte par OPUS en tableua de points, entre borne1 et borne2 en cm-1 (250 et 450 par defaut). Path est le chemin complet avec le nom du fichier. Eviter les ~/  """
    x,y=[],[]
    fs = open(Path, 'r')
    #print('Open new fic') 
#index_array = 0
    while 1: 
        txt = fs.readline()
        #print(txt)
        if ((txt =='')|(txt == '\r\n')): 
            break
        #print(txt)
        ii=-1
        while 1: # on cherche le premier espace qui limite le premier nombre
            ii = ii+1
            #print(ii)
            if ((txt[ii] == ' ') |(txt[ii] == '\t')):
                break
        
        x.append(float(txt[0:ii]))
        y.append(float(txt[ii:]))  
#        if len(txt) == 21 : #nu >= 10000 cm-1
#            x.append(float(txt[0:11]))
#            y.append(float(txt[11:]))
#        elif len(txt) == 20 : #nu >= 1000 cm-1
#            x.append(float(txt[0:10]))
#            y.append(float(txt[10:]))
#        elif len(txt) == 19 : #nu >= 100 cm-1
#            x.append(float(txt[0:9]))
#            y.append(float(txt[9:]))
#        elif len(txt) == 18 : #nu >= 10 cm-1
#            x.append(float(txt[0:8]))
#            y.append(float(txt[8:]))
#        elif len(txt) == 17 : #nu >= 1 cm-1
#            x.append(float(txt[0:7]))
#            y.append(float(txt[7:]))

        #x[index_array],y[index_array] = float(txt[0:9]),float(txt[10:17])
        #index_array = index_array+1
        
    fs.close()
    x = np.array(x)
    y = np.array(y)
    if ((borne1 == 0) & (borne2 == 0)) :
        pass    
    else :
        index_ok = ((x<borne2) & (x>borne1))
        x = x[index_ok]
        y = y[index_ok]

    return x,y
    
def Read_WinSpec(Path,borne1 = 0,borne2 = 0) :
    """ Charge un fichier provenant d'un fichier exporte par OPUS en tableua de points, entre borne1 et borne2 en cm-1 (250 et 450 par defaut). Path est le chemin complet avec le nom du fichier. Eviter les ~/  """
    x,y=[],[]
    fs = open(Path, 'r')
   
    while 1: 
        txt = fs.readline()
        if ((txt =='')|(txt == '\r\n')): 
            break
        ii=-1
        index_line=[]
        while 1: # on cherche le premier espace qui limite le premier nombre
            ii = ii+1
            if (txt[ii:ii+1] == '\t'):
                index_line.append(ii)
            if (txt[ii:ii+4] == '\r\n'):
                break
        x.append(float(txt[:index_line[0]]))
        y.append(float(txt[index_line[1]:]))  
    fs.close()
    x = np.array(x)
    y = np.array(y)
    if ((borne1 == 0) & (borne2 == 0)) :
        pass    
    else :
        index_ok = ((x<borne2) & (x>borne1))
        x = x[index_ok]
        y = y[index_ok]

    return x,y

def Read_WinSpec2(Path,borne1 = 0,borne2 = 0) :
    """ Charge un fichier provenant d'un fichier exporte par OPUS en tableua de points, entre borne1 et borne2 en cm-1 (250 et 450 par defaut). Path est le chemin complet avec le nom du fichier. Eviter les ~/  """
    x,y=[],[]
    fs = open(Path, 'r')
   
    while 1: 
        txt = fs.readline()
        if ((txt =='')|(txt == '\r\n')): 
            break
        ii=-1
        index_line=[]
        while 1: # on cherche le premier espace qui limite le premier nombre
            ii = ii+1
            if (txt[ii:ii+1] == ';'):
                index_line.append(ii)
            if (txt[ii:ii+4] == '\r\n'):
                break
        x.append(float(txt[:index_line[0]]))
        y.append(float(txt[index_line[0]+1:]))  
    fs.close()
    x = np.array(x)
    y = np.array(y)
    if ((borne1 == 0) & (borne2 == 0)) :
        pass    
    else :
        index_ok = ((x<borne2) & (x>borne1))
        x = x[index_ok]
        y = y[index_ok]

    return x,y


def Read_Spectrum2(Path,borne1 = 0,borne2 = 0) :
    """ Charge un fichier provenant d'un fichier exporte par OPUS en tableua de points, entre borne1 et borne2 en cm-1 (250 et 450 par defaut). Path est le chemin complet avec le nom du fichier. Eviter les ~/ \n
    Retourne la longueur d'onde en microns."""
    x,y=[],[]
    fs = open(Path, 'r')
    while 1: 
        txt = fs.readline()
        if txt =='': 
            break
        ii=-1
        while 1: # on cherche le premier espace qui limite le premier nombre
            ii = ii+1
            if ((txt[ii] == ' ') |(txt[ii] == '\t')):
                break
        x.append(float(txt[0:ii]))
        y.append(float(txt[ii:]))  

        
    fs.close()
    x = np.array(x)
    y = np.array(y)
    if ((borne1 == 0) & (borne2 == 0)) :
        pass    
    else :
        index_ok = ((x<borne2) & (x>borne1))
        x = x[index_ok]
        y = y[index_ok]

    return 1e4/x,y 

def Read_H2O(Path,borne1 = 250.,borne2 = 450.) :
    """ Charge un fichier provenant d'un fichier exporte par OPUS en tableua de points, entre borne1 et borne2 en cm-1 (250 et 450 par defaut). Path est le chemin complet avec le nom du fichier. Eviter les ~/  """
    x,y=[],[]
    fs = open(Path, 'r') 
#index_array = 0
    while 1: 
        txt = fs.readline()
        if txt =='': 
            break
        x.append(float(txt[2:12]))
        y.append(float(txt[14:-1]))
        
         
    fs.close()
    x = np.array(x)
    y = np.array(y)
    index_ok = ((x<borne2) & (x>borne1))
    x = x[index_ok]
    y = y[index_ok]

    return x,y 


def Plot_RCWA_Ssim(Path) :
    """ Charge un fichier provenant d'un fichier sortie Spectre simple de LaunchRCWA_v5  """
    #/Users/simonvassant/Documents/20090707_TestsPython/Result_Champ/SSim_O21Pr_2000H_res1000F1.res
    x,y=[],[]
    fs = open(Path, 'r') 
#index_array = 0
    while 1: 
        txt = fs.readline()
        if txt =='': 
            break
        x.append(float(txt[0:12]))
        y.append(float(txt[13:-1]))
        #x[index_array],y[index_array] = float(txt[0:9]),float(txt[10:17])
        #index_array = index_array+1
         
    fs.close()
    plt.figure(1)
    plt.plot(x,y)
    plt.xlabel(r"Longueur d'onde $(\mu m)$")
    plt.ylabel('R')
    
    
def Read_Rcwa_Matlab(Path) :
    """ Charge un fichier provenant d'un fichier sortie Spectre simple de LaunchRCWA_v5, retourne x,y  """    
    x,y=[],[]
    fs = open(Path, 'r') 
    while 1: 
        txt = fs.readline()
        if txt =='': 
            break
        x.append(float(txt[0:25]))
        y.append(float(txt[29:-2])) 
    fs.close()
    return x,y
    
def powerpoint_style(Axe_tick_size=15,Line_size=3) :
    """ Elargit les traits, agrandit les labels.""" 
    fig = plt.gcf()
    ax  = fig.gca()
    # trouve tous les trucs avec linewidth et les modifie
    def myfunc(x):
        return hasattr(x, 'set_linewidth')
    for o in fig.findobj(myfunc):
        o.set_linewidth(Line_size)
    # en particuliers les marqueurs des ticks
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_markersize(Line_size+2)
        line.set_markeredgewidth(Line_size)
        
    # trouve tous les textes et les modifie
    def myfunc(x):
        return hasattr(x, 'set_fontsize')
    for o in fig.findobj(myfunc):
        o.set_fontsize(Axe_tick_size)
    # les labels un peu plus larges
    ax.set_xlabel(ax.get_xlabel(),fontsize = Axe_tick_size+5,labelpad=2)
    ax.set_ylabel(ax.get_ylabel(),fontsize = Axe_tick_size+5,labelpad=2)
    
    def myfunc(x):
        return hasattr(x, 'set_markersize')
    for o in fig.findobj(myfunc):
        o.set_markersize(Line_size+4)
        
    def myfunc(x):
        return hasattr(x, 'set_markeredgewidth')
    for o in fig.findobj(myfunc):
        o.set_markeredgewidth(Line_size)
       
    
    fig.show()

def powerpoint_style2(Axe_tick_size=15,Line_size=3) :
    """ Elargit les traits, agrandit les labels.""" 
    fig = plt.gcf()
    def myfunc(x):
        return hasattr(x, 'set_linewidth')
    for o in fig.findobj(myfunc):
        o.set_linewidth(Line_size)
           
    def myfunc(x):
        return hasattr(x, 'set_markersize')
    for o in fig.findobj(myfunc):
        o.set_markersize(Line_size+4)
    def myfunc(x):
        return hasattr(x, 'set_markeredgewidth')
    for o in fig.findobj(myfunc):
        o.set_markeredgewidth(Line_size)
    for ax in fig.axes:
    
        # trouve tous les trucs avec linewidth et les modifie
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(Axe_tick_size)
            
        #for item in ([ax.xaxis.label, ax.yaxis.label]):
        #    item.set_fontsize(Axe_tick_size+5)
        for line in ax.get_xticklines() + ax.get_yticklines():
            line.set_markersize(Line_size+2)
            line.set_markeredgewidth(Line_size)

def powerpoint_style3(fig,Axe_tick_size=15,Line_size=3) :
    """ Elargit les traits, agrandit les labels.""" 
    #fig = plt.gcf()
    def myfunc(x):
        return hasattr(x, 'set_linewidth')
    for o in fig.findobj(myfunc):
        o.set_linewidth(Line_size)
           
    def myfunc(x):
        return hasattr(x, 'set_markersize')
    for o in fig.findobj(myfunc):
        o.set_markersize(Line_size+4)
    def myfunc(x):
        return hasattr(x, 'set_markeredgewidth')
    for o in fig.findobj(myfunc):
        o.set_markeredgewidth(Line_size)
    for ax in fig.axes:
    
        # trouve tous les trucs avec linewidth et les modifie
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(Axe_tick_size)
            
        #for item in ([ax.xaxis.label, ax.yaxis.label]):
        #    item.set_fontsize(Axe_tick_size+5)
        for line in ax.get_xticklines() + ax.get_yticklines():
            line.set_markersize(Line_size+2)
            line.set_markeredgewidth(Line_size) 
        
       
    
    #fig.show()
    
def differentiate1(x,y) :
    Y = np.zeros(len(x)-1)
    X = x[0:-1]
    for index in range( len(x)-1):
       Y[index] =(y[index+1]-y[index])/(x[index+1]-x[index])
        
    return np.array(X),np.array(Y)

def epsilon_GaAs_TISB(l_onde,n_2D,w12,f,Ts):
    """ n_2D en cm-2, w12 en microns, T en cm-1"""
    ev          = 1.60218e-19
    eps0        = 8.854e-12
    c           = 3e8
    
    v           = 1e4/l_onde
    
    epsinf      = 11;
    wL          = 291.2;
    wT          = 267.89;
    T           = 2.54;
    epsGaAs     = epsinf*(1+(wL**2-wT**2)/(wT**2-v**2-1j*T*v)) 
    
    meff        = 9.109e-31*0.067
    ns          = (n_2D/22e-7)*1e6 # passage de cm-2 ? m-2 On fixe l'epaisseur du puit a 20nm, ordre de grandeur...
    wp         = np.sqrt((ns*ev**2)/(meff*eps0*epsinf)) # SI
    wp          = wp*1e-2/(2*np.pi*c) #cm-1
    if w12==0:
        wp=0
    else:
        w12         = 1e4/w12                       # cm-1
    
    
    eps_ssbandes = epsinf*(f*wp**2./(w12**2-v**2-1j*Ts*v))
    eps = epsGaAs + eps_ssbandes
    return eps.real + 1j*np.abs(eps.imag)

def epsilon_fit_GaAsdope(l_onde,wL,wT,T,n,epsinf,T1 = 0):
    eps0=8.854e-12
    mel=9.109e-31
    ev=1.60218e-19
    c= 3e8

    v = 1e4/l_onde
    epsilon_GaAs = epsinf*(1+(wL**2-wT**2)/(wT**2-v**2-1j*T*v)) 
    dopage      = n #cm-3
    n3D         = n*1e6 #passage en m-3 %1cm = 1e-2m ; %1cm3 = 1e-6m3, %1cm-3 = 1e6m-3
    meffective  = mel*0.067
        
    # G.Yu et al. ASS 199 p160-165 (2002)
    if T1 == 0 :
        gammap      = 5.17e-14 + 2.23e-14*np.exp(-dopage/7.62e16) # le dopage est en cm-3 dans cette expression.
        gammap      = (1/gammap*1e-2)/(2*np.pi*c) # cm-1
    else :
        gammap      = T1

    wpSI        = sqrt_maystre((n3D*ev**2)/(eps0*epsinf*meffective)) #SI : sr.s-1
    wp          = (wpSI*1e-2)/(2*np.pi*c)# conversion cm-1  
        
    eps_elibres = epsinf*wp**2/(v**2 + 1j*v*gammap);
    if n==0 :
        epsilon = epsilon_GaAs
    else :
        epsilon = epsilon_GaAs - eps_elibres
    # Ajout d'une petite correction, je garde toujours une partie imaginaire positive.
    return epsilon.real + 1j*np.abs(epsilon.imag)

def epsilon_fit_AlGaAs(l_onde,vl1,vl2,vt1,vt2,gl1,gl2,gt1,gt2,epsinf):
    v = 1e4/l_onde    
    epsilon = epsinf*(((vl1**2)-(v**2)-1j*v*gl1)*((vl2**2)-(v**2)-1j*v*gl2))/(((vt1**2)-(v**2)-1j*v*gt1)*((vt2**2)-(v**2)-1j*v*gt2))
    
    return epsilon.real + 1j*np.abs(epsilon.imag)
    
def epsilon_fit_SiC(l_onde,wL,wT,T,epsinf) :
    v = 1e4/l_onde
    epsilon = epsinf*(1+(wL**2-wT**2)/(wT**2-v**2-1j*T*v))    
    return epsilon.real + 1j*np.abs(epsilon.imag)
       


def Plot_Spectrum(Path,borne1 = 0,borne2 = 0) :
    """ Charge un fichier provenant d'un fichier exporte par OPUS en tableua de points, entre borne1 et borne2 en cm-1 (250 et 450 par defaut). Path est le chemin complet avec le nom du fichier. Eviter les ~/  """
    x,y=[],[]
    fs = open(Path, 'r') 
#index_array = 0
    while 1: 
        txt = fs.readline()
        if txt =='': 
            break
        x.append(float(txt[0:9]))
        y.append(float(txt[10:17]))
        #x[index_array],y[index_array] = float(txt[0:9]),float(txt[10:17])
        #index_array = index_array+1
        
    fs.close()
    x = np.array(x)
    y = np.array(y)
    if ((borne1 == 0) & (borne2 == 0)) :
        pass    
    else :
        index_ok = ((x<borne2) & (x>borne1))
        x = x[index_ok]
        y = y[index_ok]
    plt.figure(1)
    plt.plot(x,y)
    plt.xlabel(r"Nombre d'onde $(cm^{-1})$")
        
def Calc_dopage_labs(l_onde) :
    """ calcule le dopage en se basant sur la longueur d'onde d'absorption, seulement valable pour GaAs"""
    eps0=8.854e-12
    mel=9.109e-31
    ev=1.60218e-19
    c= 3e8
    return (1e12*4*(np.pi*c)**2*eps0*(10.9)*(0.067*mel/(l_onde*ev)**2))*1e-6
    
    
def RMS(x):
    return sqrt_maystre(np.sum(x**2)/len(x))
    
def epsilon_GaAs_ssbandes(l_onde,V,tau) :
    """ Retourne la constante diélectrique de GaAs avec l'ajout de la contribution des transitions inter-sousbandes. l_onde en microns, V en volts, tau en s (entre 10e-12 et 0.1e-12).
    Dans l'ordre 1-2, 1-3,1-4,2-3,2-4,et 3-4. Le fichier importé comprend Vg, wdmm',wmm' (cf note Alex). V ne prend que les valeurs 0,0.3,0.4,0.6 et 0.7"""
    
    hb=1.05458e-34
    ev=1.60218e-19
    c= 3e8

    Vread,wd,wmm = [],[],[]
    fs = open('/Users/simonvassant/Documents/Matlab/SimonBox/Epsilon_transition22nm.dat','r')
    while 1: 
        txt = fs.readline()
        if txt =='': 
            break
        Vread.append(float(txt[0:3]))
        wd.append(float(txt[4:22]))
        wmm.append(float(txt[23:-1]))
    fs.close()
    
    if V == 0 :
        index = 0
    elif V == 0.3:
        index = 6
    elif V ==0.4:
        index = 12
    elif V == 0.5:
        index = 18
    elif V == 0.6:
        index = 24
    elif V == 0.7:
        index = 30
    else : 
        print ('Pas bonne valeur pour V : 0, 0.3, 0.4, 0.6, ou 0.7 uniquement')
    ksi = np.zeros((len(l_onde),6),dtype = np.complex)
    wd = np.array(wd)
    wmm = np.array(wmm)
    #print wmm
    w = 1e4/l_onde # en cm-1
    #wd = np.sin(theta*np.pi/180)*(wd*1e-5*ev)/(2*np.pi*c*hb) # en cm-1
    wd = (wd*1e-5*ev)/(2*np.pi*c*hb)
    wmm = (wmm*1e-5*ev)/(2*np.pi*c*hb) # en cm-1
    #print wmm
    #(hb*2*np.pi*c)/(l_onde*1e-9*ev) # passage microns-meV 
    gamma = (1/tau)*1e-2/(2*np.pi*c) # passage cm-1
      
    compteur = 0 
    while compteur<6 :        
        index_read = index+compteur 
        ksi[:,compteur] = wd[index_read]/(wmm[index_read] -w - 1j*gamma)
        print(str(1e4/wmm[index_read]) + ' microns')
        compteur = compteur+1
    epsilon_ssbandes = np.sum(ksi,1)
    v = 1e4/l_onde    
    epsinf = 10.9
    wL = 291.2
    wT = 267.98
    T= 2.054 #Palik Value
    epsilon_GaAs = epsinf*(1+(wL**2-wT**2)/(wT**2-v**2-1j*T*v)) 
    
    return epsilon_GaAs + epsilon_ssbandes
    
    
    
def gap_AlGaAs(x):
    """ return the AlGaAs energy gap (eV) as a function of the concentration x"""
    if type(x)==np.ndarray:
        Eg = 1.424 + 1.247*x
        Eg[x>=0.45] = 1.9+0.125*x+0.143*x**2
    else:
        if (x<=0.45):
            Eg = 1.424 + 1.247*x
        else:
            Eg = 1.9+0.125*x+0.143*x**2
    return Eg        

def Alex_res(w,wp,wr,g):
    return wp/(w-wr-1j*g)
def Alex_res2(w,wp,wr,g):
    return wp**2/(w**2-wr**2-1j*g*w)


def fact(n):
    """fact(n): calcule la factorielle de n (entier >= 0)"""
    if n<2:
        return 1
    else:
        return n*fact(n-1)
        
def plot_v(l_onde,indexes=0):
    """ Trace une ligne verticale automatiquement ajustée à l'échelle de la figure au moment ou la ligne est tracée \n
    Dans le cas ou l_onde est un simple entier, on trace à la valeur de cet entier. Si c'est un array, liste, tuple, \n
    on trace à la valeur indiquee par indexes qui est un booléen de la taille de l_onde."""
    
    ax = plt.gcf().gca()
    
    if (type(l_onde) != int):
        for index in indexes :
            plt.vlines(l_onde[index],ax.get_yticks()[0],ax.get_yticks()[-1])
    else :
        plt.vlines(l_onde,ax.get_yticks()[0],ax.get_yticks()[-1])
        
def plot_h(l_onde,indexes=0):
    """ Trace une ligne horizontale automatiquement ajustée à l'échelle de la figure au moment ou la ligne est tracée \n
    Dans le cas ou l_onde est un simple entier, on trace à la valeur de cet entier. Si c'est un array, liste, tuple,\n
     on trace à la valeur indiquee par indexes qui est un booléen de la taille de l_onde."""
    ax = plt.gcf().gca()
    if (type(l_onde) != int):
        for index in indexes :
            plt.hlines(l_onde[index],ax.get_xticks()[0],ax.get_xticks()[-1])
    else :
        plt.hlines(l_onde,ax.get_xticks()[0],ax.get_xticks()[-1])   
        
def concentration(conc,V = 50):
    """conc en mol/L, donne le poids de R6G nécessaire pour atteindre cette concentration dans V(=50 par défaut) ml de résine"""
    V = V*1e-3 #L
    m_mol = 479.02 #g/mol
    m = conc*m_mol*V # mol/L*g/mol*L = g
    print("concentration de %s mol/L dans %s ml revient à %s g de R6G" %(conc,V*1e3,m))
    return m
def sliding_Lorentz(l_onde,amp,w0,gamma):
    """ retourne une lorentzienne, l_onde en microns, amp,w0,tau en cm-1"""
    return amp/(w0 - (1e4/l_onde) - 1j*gamma)



# Physique semi-conducteur

def Eg_fct_T(Eg0,alpha,beta,T) :
    """Retourne Eg(T) en fonction de Eg(0), alpha, beta, et T"""
    return Eg0-((T*T*alpha*1e-3)/(beta+T))
        
def m_e(Ep,Eg0,F,Dso,alpha,beta,T):
    """ retourne la masse effective de bande en fonction des paramètres d'entrées : \n
        Ep,F paramètres issus de la théorie k.p de Kane \n
        Eg0 énergie de gap à 0K \n
        Dso séparation spin-orbite (spin-orbit splitting)\n
        alpha,beta : coefficient de Varshni\n
        T température en Kelvin"""
    Eg = Eg_fct_T(Eg0,alpha,beta,T)
    return 1./(1+2*F + (Ep*(Eg+Dso/3))/(Eg*(Eg+Dso)))
def m_hh_lh(g1,g2):
    """Retourne les masse effectives des trous lourds et trous léger en fonction des coefficients de Luttinger"""
    return 1./(g1-2*g2), 1./(g1 + 2*g2)
def ternary_value(A,B,C,x):
    """ retourne la valeur interpolée pour un alliage ternaire de III-V avec comme paramètres de courbure C"""
    return (1-x)*A + x*B - (1-x)*C
    
    
def Read_RperOrders(Name) :
    fid = open(Name,'r')
    R_tot = []
    while 1: 
        line = fid.readline()
        if line =='': 
            break    
        R_line = []
        sep = []
        for ii in range(len(line)):
            if line[ii]==' ': # séparateur pour ce type de fichier
                sep.append(ii)
        R_line.append(line[0:sep[0]]) # on oublie pas la premiere ligne
        for ii in range(len(sep)-1):
            if line[sep[ii]:sep[ii+1]] == ' ': # cela arrive, on négligle
                pass
            else :
                R_line.append(float(line[sep[ii]:sep[ii+1]])) # bonne valeur stockée
        
        R_line.append(line[sep[-1]:len(line)])     # On n'oubie par la dernière colonne   
        R_tot.append(R_line) # on stocke
        
    nb_order = np.int(len(R_tot[0])/4) # dans le fichier, il y a des colonnes en trop, toujours à zero. Meme nombre de colonne inutiles que de colonnes utiles donc division par deux, puis, on regroupe les colonnes deux à deux pour former les r complexes donc autre division par deux, soit par 4 au total.
    print('nb order : %s' %nb_order)
    R_array = np.ndarray((len(R_tot),nb_order),dtype=complex)
    
    for ii in range(len(R_tot)):
        index = 0
        for iii in range(nb_order):
            R_array[ii,iii] = float(R_tot[ii][index]) + 1j*float(R_tot[ii][index+1])
            index = index+2        
    fid.close()
    return R_array    
    
    
# from: http://www.cs.princeton.edu/introcs/21function/ErrorFunction.java.html
# Implements the Gauss error function.
#   erf(z) = 2 / sqrt(pi) * integral(exp(-t*t), t = 0..z)
#
# fractional error in math formula less than 1.2 * 10 ^ -7.
# although subject to catastrophic cancellation when z in very close to 0
# from Chebyshev fitting formula for erf(z) from Numerical Recipes, 6.2
def erf(z):
        t = 1.0 / (1.0 + 0.5 * abs(z))
        # use Horner's method
        ans = 1 - t * np.exp( -z*z -  1.26551223 +
                                                t * ( 1.00002368 +
                                                t * ( 0.37409196 + 
                                                t * ( 0.09678418 + 
                                                t * (-0.18628806 + 
                                                t * ( 0.27886807 + 
                                                t * (-1.13520398 + 
                                                t * ( 1.48851587 + 
                                                t * (-0.82215223 + 
                                                t * ( 0.17087277))))))))))
        if z >= 0.0:
                return ans
        else:
                return -ans  
                
                
                
def GaAs_Flat(p):
    borne1  = p[0]
    borne2  = p[1]
    theta   = p[2]
    Polar   = 'TM'
    Folder_m = '20100504_FTIR_P35F24'
    Folder_r = Folder_m
    
    Ech_name        = 'P35F24Flat'+Polar+'_'+  str(theta) + 'dg_mac.dpt'
    Ref_name        = 'Au'+Polar+'_'+  str(theta) + 'dg_mac.dpt'

    Path            = '/Users/simonvassant/Documents/' + Folder_m +'/'+ Ech_name 
    l_onde_cm,S_cm  = Read_Spectrum(Path,borne1 ,borne2)
    l_onde,S        = convert_cm_microns(l_onde_cm,S_cm)
    Path            = '/Users/simonvassant/Documents/'+ Folder_r +'/'+ Ref_name
    l_onde,Ref      = Read_Spectrum(Path,borne1,borne2)
    l_onde,Ref      = convert_cm_microns(l_onde,Ref)
    S               =  S/Ref
    
    return l_onde,S
    
def RD_perOrders(axe_theta,axe_lambda,data):
    """ Remet sous forme de matrice data, provenant de Read_RperOrders dans le cas d'une boucle secondaire (en theta)"""  
    
    Obj = np.ndarray((axe_theta.shape[0],axe_lambda.shape[0]))
    for ii in range(len(axe_theta)):
        debut = ii*axe_lambda.shape[0]
        fin = debut+axe_lambda.shape[0]
        Obj[ii,:] = data[debut:fin]
    return Obj

def Find_Max_nointerp(x,y,minval=0.5):
    """ retourne les maximums locaux (inferieur a minval) pour une courbe y ayant comme abscisse x"""
    index          = np.arange(x.shape[0])
    yy = y.copy()
    yy[yy>minval] = np.NAN
    #yy              = 1-yy               
    yy              = np.diff(yy)                         # on derive
    yy              = np.sign(yy)                        # on recupp le signe de la derivee
    idx            = np.diff(yy)<0                       # on reccup les point pour lesquels la derivee change de signe
    index          = index[idx]
    index          = index+1                              # la diffrentiation décale d'un point vers la droite
    pics_pos,pics_value       = x[index],y[index]              # longueur d'onde des pics
    return pics_pos,pics_value

def Find_Max_nointerp2(x,y,minval=0, maxval=1):
    """ retourne les maximums locaux (inferieur a minval) pour une courbe y ayant comme abscisse x"""
    index          = np.arange(x.shape[0]-2)
    yy = y.copy()
    yy[yy<minval] = np.NAN
    yy[yy>maxval] = np.NAN
    #yy              = 1-yy               
    yy              = np.diff(yy)                         # on derive
    yy              = np.sign(yy)                        # on recupp le signe de la derivee
    idx            = np.diff(yy)<0                       # on reccup les point pour lesquels la derivee change de signe
    index          = index[idx]
    index          = index+1                              # la diffrentiation décale d'un point vers la droite
    pics_pos,pics_value       = x[index],y[index]              # longueur d'onde des pics
    return pics_pos,pics_value

def epsilon_fit_3_5(l_onde,wL,wT,TL,TT,epsinf) :
    v       = 1e4/l_onde
    epsilon = epsinf*((wL**2-v**2-1j*TL*v)/(wT**2-v**2-1j*TT*v))
    #epsilon = epsinf*(1+(wL**2 - wT**2)/(wT**2-v**2-1j*TT*v))
    return (epsilon.real + 1j*np.abs(epsilon.imag))

def epsilon_fit_Chang(l_onde,vl1,vl2,vt1,vt2,gl1,gl2,gt1,gt2,f_l1,f_l2,f_t1,f_t2,epsinf1,epsinf2):
    """ retourne les valeurs de epsx et epsz selon les formules de Chang, avec 1 pour GaAs et 2 pour AlAs"""
    # Chang PRB38 12369
    v       = 1e4/l_onde
    
    epsx = (epsinf1+epsinf2)/2 - (f_t1*(vl1**2 - vt1**2))/(-vt1**2 + v**2 + 1j*v*gt1) - (f_t2*(vl2**2 - vt2**2))/(-vt2**2 + v**2 + 1j*v*gt2)
    epsz = 1/(((1/2)*(1/epsinf1 + 1/epsinf2)) + (f_l1*(vl1**2 - vt1**2))/(-vl1**2 + v**2 + 1j*v*gl1) + (f_l2*(vl2**2 - vt2**2))/(-vl2**2 + v**2 + 1j*v*gl2))
    
#    eps1 = epsinf1*(1 - (f_t1*(vl1**2 - vt1**2))/(vt1**2 - v**2 - 1j*v*gt1))
#    eps2 = epsinf2*(1 - (f_t2*(vl2**2 - vt2**2))/(vt2**2 - v**2 - 1j*v*gt2))
#    epsx = (1/2)*(eps1+eps2)
#    epsz = 1/((1/2)*(1/eps1 + 1/eps2))
    
    return (epsx.real + 1j*np.abs(epsx.imag)),(epsz.real + 1j*np.abs(epsz.imag))
    
def epsilon_fit_Chang_homemade(l_onde,vl1,vl2,vt1,vt2,gl1,gl2,gt1,gt2,f_t1,f_t2,f_l1,f_l2,epsinf1,epsinf2):
    """ retourne les valeurs de epsx et epsz selon les formules de Chang, avec 1 pour GaAs et 2 pour AlAs"""
    # Chang PRB38 12369
    v       = 1e4/l_onde
    
    epsx = (epsinf1+epsinf2)/2 - (f_l1*(vl1**2 - vt1**2))/(-vt1**2 + v**2 + 1j*v*gt1) - (f_l2*(vl2**2 - vt2**2))/(-vt2**2 + v**2 + 1j*v*gt2)
    epsz = 1/(1/((epsinf1+epsinf2)/2) + (f_l1*(vl1**2 - vt1**2))/(-vt1**2 + v**2 + 1j*v*gt1) + (f_l2*(vl2**2 - vt2**2))/(-vt2**2 + v**2 + 1j*v*gt2))
    #epsx = (1/2)*(eps1+eps2)
    #epsz = 1/((1/2)*(1/eps1 + 1/eps2))
    
    #epsx = (1/2)*epsinf1*(1-(f_t1*(v**2 - vl1**2 + 1j*v*gl1)/(v**2 - vt1**2 + 1j*v*gt1))-\
    #                     (f_t2*(v**2 - vl2**2 + 1j*v*gl2)/(v**2 - vt2**2 + 1j*v*gt2)))
    #epsz = 1/((1/2)*(1/epsinf2)*(1+(f_l1*(v**2 - vt1**2 +1j*v*gl1)/(v**2 - vl1**2 +1j*v*gl1))+\
    #                    (f_l2*(v**2 - vt2**2 +1j*v*gl2)/(v**2 - vl2**2 +1j*v*gl2))))
    return (epsx.real + 1j*np.abs(epsx.imag)),(epsz.real + 1j*np.abs(epsz.imag))
    
def epsilon_fit_GaAsdope2(l_onde,p):
    """ p contient wL,wT,T,epsinf,n,T1"""
    wL      = p[0]
    wT      = p[1]
    T       = p[2]
    epsinf  = p[3]
    n       = p[4]
    T1      = p[5]
    
    eps0    =8.854e-12
    mel     =9.109e-31
    ev      =1.60218e-19
    c       = 3e8

    v       = 1e4/l_onde
    epsilon_GaAs = epsinf*(1+(wL**2-wT**2)/(wT**2-v**2-1j*T*v)) 
    dopage      = n #cm-3
    n3D         = n*1e6 #passage en m-3 %1cm = 1e-2m ; %1cm3 = 1e-6m3, %1cm-3 = 1e6m-3
    meffective  = mel*0.067
        
    # G.Yu et al. ASS 199 p160-165 (2002)
    if T1 == 0 :
        gammap      = 5.17e-14 + 2.23e-14*np.exp(-dopage/7.62e16) # le dopage est en cm-3 dans cette expression.
        gammap      = (gammap*1e-2)/(2*np.pi*c) # cm-1
    else :
        gammap      = T1

    wpSI        = sqrt_maystre((n3D*ev**2)/(eps0*epsinf*meffective)) #SI : sr.s-1
    wp          = (wpSI*1e-2)/(2*np.pi*c)# conversion cm-1  
        
    eps_elibres = epsinf*wp**2/(v**2 + 1j*v*gammap);
    if n==0 :
        epsilon = epsilon_GaAs
    else :
        epsilon = epsilon_GaAs - eps_elibres
    # Ajout d'une petite correction, je garde toujours une partie imaginaire positive.
    return epsilon.real + 1j*np.abs(epsilon.imag)
    
def epsilon_fit_AlGaAs2(l_onde,p):
    """ p contient vl1,vl2,vt1,vt2,gl1,gl2,gt1,gt2,epsinf"""
    vl1     = p[0]
    vl2     = p[1]
    vt1     = p[2]
    vt2     = p[3]
    gl1     = p[4]
    gl2     = p[5]
    gt1     = p[6]
    gt2     = p[7]
    epsinf  = p[8]
    v       = 1e4/l_onde    
    epsilon = epsinf*(((vl1**2)-(v**2)-1j*v*gl1)*((vl2**2)-(v**2)-1j*v*gl2))/(((vt1**2)-(v**2)-1j*v*gt1)*((vt2**2)-(v**2)-1j*v*gt2))
    
    epsilon.imag = np.abs(epsilon.imag)
    return epsilon
def epsilon_fit_GaAsssbandes(l_onde,wL,wT,T,epsinf,w0,f_o,n2,T2,gssb):
    """ On renvoie la constante diélectrique de GaAs, modifiée par le dopage,et une transition inter-sousbande à la fréquence w0\n
    On rentre wL,wT,T,w0, T2,tau en cm-1"""
    
    eps0    =8.854e-12
    mel     =9.109e-31
    ev      =1.60218e-19
    c       = 3e8
    v       = 1e4/l_onde
    epsilon_GaAs = epsinf*(1+(wL**2-wT**2)/(wT**2-v**2-1j*T*v)) 
    n3D         = n2*1e6 #passage en m-3 %1cm = 1e-2m ; %1cm3 = 1e-6m3, %1cm-3 = 1e6m-3
    meffective  = mel*0.067

    wpSI        = sqrt_maystre((n3D*ev**2)/(eps0*epsinf*meffective)) #SI : sr.s-1
    wp          = (wpSI*1e-2)/(2*np.pi*c)# conversion cm-1  
    w0          = 1e4/w0     
    #eps_elibres = epsinf*wp**2/(v**2 + 1j*v*T2);
    
    eps_tissb = (epsinf*wp**2*f_o)/((w0**2-v**2-1j*gssb*v))
    
    return (epsilon_GaAs + eps_tissb)
     
def Read_champ(Path):
    fs = open(Path,'r')
    R_tot = []
    # ici R_tot = 287*321
    while 1: 
        line = fs.readline()
        if line =='': 
            break    
        R_line = []
        sep = []
        for ii in range(len(line)):
            if line[ii]==' ': # séparateur pour ce type de fichier
                sep.append(ii)
        
        for ii in range(len(sep)-1):
            if line[sep[ii]:sep[ii+1]] == ' ': # cela arrive, on négligle
                pass
            else :
                R_line.append(float(line[sep[ii]:sep[ii+1]])) # bonne valeur stockée
        
        R_line.append(line[sep[-1]:len(line)])     # On n'oubie par la dernière colonne   
        R_tot.append(R_line) # on stocke
        if len(R_tot) == 256 :
            pass
    nb_col = np.int((len(R_tot[0])-1)/2) 
    
    R_array = np.ndarray((len(R_tot),nb_col),dtype=complex)
    
    Axe_z = np.ndarray(len(R_tot),)
    for ii in range(len(R_tot)):
        index = 1
        for iii in range(nb_col):
            if iii == 0 :
                Axe_z[ii] = R_tot[ii][iii]
            else : 
                R_array[ii,iii-1] = float(R_tot[ii][index]) + 1j*float(R_tot[ii][index+1])
                index = index+2        
    fs.close()
    return R_array,Axe_z
    
def ZCE_GaAs(n,V):
    """ n en cm-3, V en Volts"""
    eps0 = 8.85e-12 #F/m
    epsinf = 11
    q=1.60218e-19
    
    n = n*1e6 # passage en m-3
    return np.sqrt((2*epsinf*eps0/q)*(1/n)*V)    
def Plot_Champ(R_array,P,F,Axe_z,composante,l_onde,type_structure,N) :

    Zdisc = 64
    
    R_toplot = R_array
    #R_toplot = (R_array*R_array.conjugate())
    Axe_x = np.linspace(0,P,len(R_toplot[0]))
    
    Axe_z = (-Axe_z*1e6) - np.min(Axe_z)
    X,Y = np.meshgrid(Axe_x,Axe_z)
    plt.figure()
    plt.contour(X,Y,R_toplot.real,N)
    #plt.clim(0,5)
    plt.contourf(X,Y,R_toplot.real,N)
    plt.xlabel(r'x ($\mu m$)',labelpad=2)
    plt.ylabel(r'z ($\mu m$)',labelpad=2)
    plt.colorbar()
    
    
    nb_couche = np.int(R_array.shape[0]/Zdisc) # ou 32 est le Z_discparam
    LimH = np.ndarray((len(Axe_x),nb_couche))
    
    for ii in range(3) :
        LimH[:,ii] = Axe_z[((ii+1)*Zdisc)]*np.ones(Axe_x.shape)
        if ((ii==0)|(ii==1)):
            LimH[Axe_x>(P*F),ii] = np.NaN
        else :   
            LimH[Axe_x<(P*F),ii] = np.NaN
#        if ii == 2: # troisième couche, limite Ti/GaAs
#            LimH[Axe_x<(P*F)] = np.NaN
        plt.plot(Axe_x,LimH[:,ii],'k',lw=2)
#    
#    # Limites verticale
    LimV = P*F*np.ones(Axe_z.shape)
    LimV[0:Zdisc] = np.NaN
    LimV[3*Zdisc:] = np.NaN
    plt.plot(LimV,Axe_z,'k',lw=2)
    #plt.clim(0,5)
    # Définition de la geometrie
    if type_structure == 'Res_GaAs':
        # Limites horizontales
        nb_couche = np.int(R_array.shape[0]/Zdisc) # ou 32 est le Z_discparam
        LimH = np.ndarray((len(Axe_x),nb_couche))
        
        for ii in range(nb_couche-1) :
            LimH[:,ii] = Axe_z[((ii+1)*Zdisc)]*np.ones(Axe_x.shape)
            if ii==0:
                LimH[Axe_x>(P*F),ii] = np.NaN
            else :   
                LimH[Axe_x<(P*F),ii] = np.NaN
    #        if ii == 2: # troisième couche, limite Ti/GaAs
    #            LimH[Axe_x<(P*F)] = np.NaN
            plt.plot(Axe_x,LimH[:,ii],'k',lw=2)
    #    
    #    # Limites verticale
        LimV = P*F*np.ones(Axe_z.shape)
        LimV[0:Zdisc] = np.NaN
        LimV[2*Zdisc:] = np.NaN
        plt.plot(LimV,Axe_z,'k',lw=2)
    
    elif type_structure  == 'Res_GaAs_pied':
    
        nb_couche = np.int(R_array.shape[0]/Zdisc) 
        LimH = np.ndarray((len(Axe_x),nb_couche-1)) 
        for ii in range(nb_couche-1) :
            LimH[:,ii] = Axe_z[((ii+1)*Zdisc)]*np.ones(Axe_x.shape)
        # Définition de la parabole, a reemplir en dur... voir dans Full_RCWA_09DE03Res1_order.py
        
        fleche                      = 0.6
        y_p                         = np.linspace(fleche,0.01,12) # Hauteur et nb couches (Attention au sens !
        posf                        = P*(1-F)/2. + P*F
        a                           = fleche/((P*F-posf)**2)
         # on commence à la deuxième ligne horizontale
         
        LimV = np.ndarray((len(Axe_z),2*len(y_p)+1))
        LimV[:,0] = P*F*np.ones(Axe_z.shape)
        Fp1_old=F
        Fp2_old = 1
        LimH[Axe_x>(P*F),0] = np.NaN
        
        plt.plot(Axe_x,LimH[:,0],'k',lw=2)
        LimV[0:Zdisc,0] = np.NaN
        LimV[2*Zdisc:,0] = np.NaN
        plt.plot(LimV[:,0],Axe_z,'k',lw=2) 
        ii=1     
        for h_p in y_p[1:] :
            Fp1 =(-np.sqrt(h_p/a) + posf)/P
            Fp2 =(np.sqrt(h_p/a) + posf)/P
            indexes = ((Axe_x<Fp1_old*P) | ((Axe_x>P*Fp1) & (Axe_x<P*Fp2)) | (Axe_x>P*Fp2_old))
            LimH[indexes,ii] = np.NaN
            plt.plot(Axe_x,LimH[:,ii],'k',lw=2)
            if ii == len(y_p):
                pass
            else:
                LimV[:,ii] = P*Fp1*np.ones(Axe_z.shape)
                LimV[0:(ii+1)*Zdisc,ii] = np.NaN
                LimV[(ii+2)*Zdisc:,ii] = np.NaN
                
                LimV[:,2*len(y_p)-(ii-1)] = P*Fp2*np.ones(Axe_z.shape)                
                LimV[0:(ii+1)*Zdisc,2*len(y_p)-(ii-1)] = np.NaN
                LimV[(ii+2)*Zdisc:,2*len(y_p)-(ii-1)] = np.NaN
                
            plt.plot(LimV[:,ii],Axe_z,'k',lw=2) 
            plt.plot(LimV[:,2*len(y_p)-(ii-1)],Axe_z,'k',lw=2) 
            Fp1_old = Fp1
            Fp2_old = Fp2
            ii = ii+1
    
        LimH[((Axe_x<P*Fp1) | (Axe_x>P*Fp2)),-1] = np.NaN
        plt.plot(Axe_x,LimH[:,-1],'k',lw=2)  
                
            

        
        
        
    plt.ylim(Axe_z[-1],Axe_z[0])
    powerpoint_style()
    #plt.title((composante + 'P'+str(P)+'F'+str(F)+'L'+str(l_onde)))
    #plt.savefig((composante + 'P'+str(P)+'F'+str(F)+'L'+str(l_onde)+'.pdf'),dpi=150,format = 'pdf')
def Plot_Champ_ZCE(R_array,P,F,Axe_z,composante,l_onde,n) :
    ZCE = ZCE_GaAs(n,0.7)
    Zdisc = 64
    
    R_toplot = np.log(R_array*R_array.conjugate())
    Axe_x = np.linspace(0,P,len(R_toplot[0])-1)
    Axe_z = (-Axe_z*1e6) - np.min(Axe_z)
    
    plt.figure()
    plt.pcolor(Axe_x,Axe_z,R_toplot.real)
    plt.xlabel(r'x ($\mu m$)')
    plt.ylabel(r'z ($\mu m$)')
    plt.colorbar()
    powerpoint_style()
    plt.clim(0,5)
    # Définition de la geometrie
    
    # Limites horizontales
    nb_couche = np.int(R_array.shape[0]/Zdisc) # ou 32 est le Z_discparam
    LimH = np.ndarray((len(Axe_x),nb_couche))
    
    for ii in range(nb_couche-1) :
        LimH[:,ii] = Axe_z[((ii+1)*Zdisc)]*np.ones(Axe_x.shape)
        if ((ii ==0) | (ii==1)|(ii==2)|(ii==3)): # premiere couche limite Au/Vide
            LimH[Axe_x>(P*F)] = np.NaN
        plt.plot(Axe_x,LimH[:,ii],'k',lw=2)
    
    # Limites verticale
    LimV = np.ndarray((3,len(Axe_z)))
    
    for ii in range(3):
        if ii==0:
            x_lim = ZCE*1e6
        if ii ==1:
            x_lim = F*P-ZCE*1e6
        if ii==2:
            x_lim = F*P
        LimV[ii,:] = x_lim*np.ones((len(Axe_z)))
        
    LimV[:,0:Zdisc] = np.NaN
    LimV[0,0:4*Zdisc] = np.NaN
    LimV[1,0:4*Zdisc] = np.NaN
    LimV[:,5*Zdisc:] = np.NaN
    for ii in range(3):
        plt.plot(LimV[ii,:],Axe_z,'k',lw=2)
    plt.ylim(Axe_z[-1],Axe_z[0])
    plt.title((composante + 'P'+str(P)+'F'+str(F)+'L'+str(l_onde)))
    #plt.savefig((composante + 'P'+str(P)+'F'+str(F)+'L'+str(l_onde)+'.pdf'),dpi=150,format = 'pdf')
def Read_abs(Path):
    Abs_ = []
    fs = open(Path,'r')
    while 1: 
        line = fs.readline()
        if line =='': 
            break    
        Abs_.append(float(line))
    return Abs_

def Read_abs_LapsusV1(Path):
    Abs_ = []
    fs = open(Path,'r')
    ii = 0
    while 1: 
        line = fs.readline()
        if line =='': 
            break
        if ii<=7:
            Abs_.append(0)
        else :    
            Abs_.append(float(line))
        ii = ii+1
    return Abs_


def Schottky_B(S,T,I,V=0,n_i=1):
    """ Donne le potentiel Schottky en fonction du courant d'obscurité mesuré, pour un contact schottky idéal. S est la surface du contact (en m^2), et T la température en Kelvin, donne aussi si l'on veut le courant direct pour un potentiel donné. On peut préciser le facteur d'idéalité désiré, n_i. Dans le cas d'une diode idéale, il est égal à 1. Dans la réalité, il est supérieur à 1."""
    ev = 1.60218e-19
    kb = 1.380e-23
    Ar = 120 # A.cm^-2.K^-2
    S = S*1e4 # passage en cm^2   
    
    Vs = np.log(I/(S*Ar*T**2*0.067))*(-kb*T/ev)
    if V==0:
        return Vs
    else :
        return Vs,I*(np.exp(ev*V/(n_i*kb*T)) -1)
    
    
def gamma_GaAs_dope(dopage):
    """ Retourne le terme d'amortissement des électrons dans GaAs dopé, selon l'article de Yu et al. en cm-1\n Le dopage en entrée est en cm-3."""
    c=3e8
    gammap      = 5.17e-14 + 2.23e-14*np.exp(-dopage/7.62e16) # le dopage est en cm-3 dans cette expression.
    gammap      = (1/gammap*1e-2)/(2*np.pi*c) # cm-1
    return gammap
    
    
    
def Read_reticolo(Name):
    """Lit un fichier sauvé depuis Matlab et calculé avec réticolo. Le format est fixé dans matlab, retourne longueur d'onde en microns, et R de la structure.\n
    Name contient le chemin d'accès complet"""
    
    fid = open(Name,'r')
    R,l_onde = [],[]
    while 1: 
        line = fid.readline()
        if line =='': 
            break    
        l_onde.append(float(line[0:line.find('\t')]))
        R.append(float(line[line.find('\t')+1:]))
    
    return(np.array(l_onde),np.array(R))


def Read_VerticalWell(Path_name) :
    fs = open(Path_name,'r')
    L,E_tot = [],[]
    while 1:
        sep = [] 
        line = fs.readline()
        if line =='': 
            break    
        if line[0] == '#': # commentaires
            pass
        else :
            E_line=[]
            for ii in range(len(line)):
                if line[ii:ii+2]=='  ': # séparateur pour ce type de fichier
                    sep.append(ii)
                    
            L.append(float(line[0:sep[0]]))
            
            for ii in range(len(sep)-1):
                E_line.append(float(line[sep[ii]:sep[ii+1]])) # bonne valeur stockée  
            
            E_line.append(float(line[sep[-1]:]))
            E_tot.append(E_line) # on stocke 
    
    fs.close()
    # il faut réarranger
    E_tot_array = np.ndarray((len(L),len(E_tot[0])))
    E_tot_array = E_tot_array*np.NaN
    for i in range(len(E_tot)):
        for ii in range(len(E_tot[i])):
            E_tot_array[i,ii] = E_tot[i][ii]
    return L,E_tot_array

def Read_Hall(Complete_Path) : 
    fid = open(Complete_Path,'r')
    L,A = [],[]
    while 1:
        sep = [] 
        line = fid.readline()
        if line =='': 
            break    
        if line[0] == '#': # commentaires
            pass
        elif line[0] == '@':
            pass
        else :
            E_line=[]
            for ii in range(len(line)):
                if line[ii:ii+3]=='   ': # séparateur pour ce type de fichier
                    sep.append(ii)
                    
            L.append(float(line[0:sep[0]]))
            
            for ii in range(len(sep)-1):
                E_line.append(float(line[sep[ii]:sep[ii+1]])) # bonne valeur stockée  
            
            E_line.append(float(line[sep[-1]:]))
            A.append(E_line) # on stocke 
    
    fid.close()    
    return np.array(L),np.array(A)

def Read_RMCA_out(Complete_Path):
    """ Retourne L (longueur d'onde (microns)) et R (Reflectivite)"""
    fid = open(Complete_Path,'r')
    L,R = [],[]
    while 1: 
        line = fid.readline()
        if line =='': 
            break    
        else :
            L.append(float(line[:25]))
            R.append(float(line[27:-2]))
    return np.array(L),np.array(R)
    
def Read_RMCA_basic(Complete_Path):
    """ Retourne L (longueur d'onde (microns)) et R (Reflectivite)"""
    fid = open(Complete_Path,'r')
    S = []
    while 1: 
        line = fid.readline()
        if line =='': 
            break    
        else :
            S.append(float(line))
            #R.append(float(line[27:-2]))
    return np.array(S)    

def print_fit_param(Nom):
    
    fid = open(Nom,'r')
    A = pk.load(fid)
    fid.close()
    # par défaut
    param_name = ['n1',      'T1',     'n2',     'T2',     'f_o',    'gssb',\
                            'Polar',      'type_struct',      'theta',  \
                            'vl1_sr', 'vl2_sr', 'vt1_sr', 'vt2_sr', 'gl1_sr', 'gl2_sr', 'gt1_sr', 'gt2_sr', 'epsinf_sr', \
                            'vl1_b',  'vl2_b',  'vt1_b',  'vt2_b',  'gl1_b',  'gl2_b',  'gt1_b',  'gt2_b',  'epsinf_b', \
                            'hc_1','hc_2','hc_2','hc_4','Anisotropie','wL',     'wT',     'T',      'epsinf','w0', \
                            'P','F','H_res','ordre','Vapp']

    angle = A.keys()[0][:2]
    ii = 0
    for ii in range(A[angle+'_param'].shape[0]):
        if A[angle+'_scalederrors'][ii] !=0 :
            print (param_name[ii] + ' = %s +/- %s vs %s' %(A[angle+'_param'][ii],A[angle+'_scalederrors'][ii],A[angle+'init'][ii]))    

def convert_dop_wp(n):
    """ give the plasma frequency in s-1 for a given doping level n (cm-3) in Gaas"""
    eps0    =8.854e-12
    me     =9.109e-31
    e      =1.60218e-19
    epsinf = 11
    n = n*1e6 #convertion en m-3
    return (np.sqrt((n*e**2)/(eps0*0.067*me*epsinf)))/(2*np.pi)

def convert_dop_cmm1(n,fracme=0.067,epsinf=11):
    """ give the plasma frequency in s-1 for a given doping level n (cm-3) in Gaas"""
    eps0    =8.854e-12
    me     =9.109e-31
    e      =1.60218e-19
    #epsinf = 11
    n = n*1e6 #convertion en m-3
    c = 3e8
    return 1e-2*((np.sqrt((n*e**2)/(eps0*fracme*me*epsinf))))/(2*np.pi*c)

def convert_wpcmm1_dop(n):
    eps0    =8.854e-12
    me     =9.109e-31
    e      =1.60218e-19
    epsinf = 11
    #n = n*1e6 #convertion en m-3
    c = 3e8
    return (4*eps0*epsinf*0.067*me*(np.pi*c*n)**2)/(1e2*e**2)
    
    
def Read_Petru(Path,borne1=0,borne2=0):
    fid = open(Path,'r')
    L = []
    M = []
    while 1: 
        line = fid.readline()
        if line =='': 
            break
                
        if line == '\r\n':
            pass
        else:
            L.append(float(line[:line.find('\t')]))
            M.append(float(line[line.find('\t')+1:]))
    fid.close()
    L = np.array(L)
    M = np.array(M)
    if ((borne1 == 0) & (borne2 == 0)) :
        pass    
    else :
        index_ok = ((L<borne2) & (L>borne1))
        L = L[index_ok]
        M = M[index_ok]
    return L,M
def epsilon_fit_3Lorentz(l_onde,vl1,vl2,vl3,vt1,vt2,vt3,gl1,gl2,gl3,gt1,gt2,gt3,epsinf):
    v = 1e4/l_onde
    return epsinf*((vl1**2 - v**2 -1j*v*gl1)/(vt1**2 - v**2 -1j*v*gt1))*((vl2**2 - v**2 -1j*v*gl2)/(vt2**2 - v**2 -1j*v*gt2))*((vl3**2 - v**2 -1j*v*gl3)/(vt3**2 - v**2 -1j*v*gt3))
    
def convert_THz_microns(x):
    return 1e-12/convert_Gamma_cm_Tau_s(1e4/x)

def Planck_w(w,T):
    """ w en rad.s-1, T en kelvin, retourne la loi de Planck : luminance du corps noir"""
    c=3e8
    hb      =1.05458e-34
    kb = 1.380e-23
    return  (1./(4*np.pi**3*c**2))*(hb*w**3)/(np.exp(((hb*w)/(kb*T))-1))
def Planck_lambda(l_onde,T):
    c = 3e8
    h  =6.626068e-34
    kb = 1.380e-23
    return (2*h*c**2)/(l_onde**5*(np.exp(h*c/(kb*l_onde*T)) -1))

def Wien(T):
    """Selon la loi de deplacement de Wien :retourne le maximum (microns) de Planck en fonction de la temperature (K)"""
    return (2898/T)

def Stefan(T):
    """retourne la luminance totale (W.m-2.K-4) d'un corps à une temperature T (Kelvin). la cste de Stefan est donnee par sigma=pi^2kb^4/60hbar^3c^2 = 5.67e-8."""
    return 5.67e-8*T**4/np.pi 
def Read_Palik():
    """ Retourne L (longueur d'onde (microns)) et epsilon"""
    fid = open('/Users/simonvassant/TheseSave/Matlab_RMCA/RMCA_Fab_CuN17_Champ/Au_Palik.dat','r')
    L,eps = [],[]
    while 1: 
        line = fid.readline()
        if line =='': 
            break    
        else :
            L.append(float(line[:25]))
            eps.append(float(line[27:54])+1j*float(line[55:-2]))
    return np.array(L),np.array(eps)
    
def Read_Ag_Palik():
    fid = open('/Volumes/Stock/PostDoc_Erlangen/DocMatos/Materials_Data_Base/Ag_Palik.dat','r')
    L,eps = [],[]
    while 1: 
        line = fid.readline()
        if line =='': 
            break    
        else :
            ii = -1
            index_line=[]
            while 1: # on cherche le premier espace qui limite le premier nombre
                ii = ii+1 
                if (line[ii:ii+2] == '  '):
                    index_line.append(ii)
                if (line[ii:ii+2] == '\n'):
                    break
            L.append(float(line[index_line[1]:index_line[2]]))
            n = float(line[index_line[2]:index_line[3]])
            k = float(line[index_line[3]:])
            eps.append((n+1j*k)**2)
    #eps = np.array(eps)
    return np.array(L),np.array(eps)
    
def Read_Al_Palik():
    """ Retourne L (longueur d'onde (microns)) et epsilon"""
    fid = open('/Volumes/Stock/PostDoc_Erlangen/DocMatos/Materials_Data_Base/Al_from_Palik.dat','r')
    L,eps = [],[]
    while 1: 
        line = fid.readline()
        if line =='': 
            break    
        else :
            L.append(float(line[:25]))
            eps.append(float(line[27:54])+1j*float(line[55:-2]))
    return np.array(L),np.array(eps)


def Read_AuJC():
    """ Retourne L (longueur d'onde (microns)) et epsilon"""
    fid = open('/Users/simonvassant/Documents/Matlab_RMCA/RMCA_Fab_CuN17_Champ/AuJC.dat','r')
    L,n,k = [],[],[]
    while 1: 
        line = fid.readline()
        if line =='': 
            break    
        else :
            sep = []
            for ii in range(len(line)):
                if line[ii:ii+4]=='    ': # séparateur pour ce type de fichier
                    sep.append(ii)
            
            L.append(float(line[:sep[0]]))
            n.append(float(line[sep[0]:sep[1]]))
            k.append(float(line[sep[1]:sep[2]]))
    fid.close()
    L = np.array(L)
    L = convert_mev_microns(L*1e3)
    n = np.array(n)
    k = np.array(k)
    eps = (n+1j*k)**2
    return np.array(L),np.array(eps)

def Read_Palik_Al2O3_a():
    """ Retourne L (longueur d'onde (microns)) et epsilon"""
    fid = open('/Users/simonvassant/TheseSave/PaliK_Opticalconstant/Al2O3_B_Palik.dat','r')
    L,eps = [],[]
    while 1: 
        line = fid.readline()
        if line =='': 
            break    
        else :
            L.append(float(line[18:25]))
            eps.append((float(line[26:34])+1j*float(line[35:40]))**2)
    return np.array(L),np.array(eps)

def Read_Palik_Al2O3_b():
    """ Retourne L (longueur d'onde (microns)) et epsilon"""
    fid = open('/Users/simonvassant/TheseSave/PaliK_Opticalconstant/Al2O3_B_Palik.dat','r')
    L,eps = [],[]
    while 1: 
        line = fid.readline()
        if line =='': 
            break    
        else :
            L.append(float(line[18:25]))
            eps.append((float(line[59:65])+1j*float(line[66:73]))**2)
    return np.array(L),np.array(eps)
def Read_Palik_Al2O3_c():
    """ Retourne L (longueur d'onde (microns)) et epsilon"""
    fid = open('/Users/simonvassant/TheseSave/PaliK_Opticalconstant/Al2O3_B_Palik.dat','r')
    L,eps = [],[]
    while 1: 
        line = fid.readline()
        if line =='': 
            break    
        else :
            L.append(float(line[18:25]))
            eps.append((float(line[42:49])+1j*float(line[50:57]))**2)
    return np.array(L),np.array(eps)
def Read_Palik_Al2O3_d():
    """ Retourne L (longueur d'onde (microns)) et epsilon"""
    fid = open('/Users/simonvassant/TheseSave/PaliK_Opticalconstant/Al2O3_B_Palik.dat','r')
    L,eps = [],[]
    while 1: 
        line = fid.readline()
        if line =='': 
            break    
        else :
            L.append(float(line[18:25]))
            eps.append((float(line[74:81])+1j*float(line[82:]))**2)
    return np.array(L),np.array(eps)
    

    
def delta_z(l_onde,epsilon,theta):
    k0 = 2*np.pi/(l_onde*1e-6)
    kz = np.sqrt(epsilon*k0**2 - (k0*np.sin(theta*np.pi/180))**2)
    return 1/(2*kz.imag)

def CuN17vol(n,h,w,V):
    """n nb porteurs (cm-3), h hauteur (nm), w largeur (nm), V=potentiel surface (eV)\n
    Retourne la surface de la coquille et du coeur en cm-2."""
    Vshell = 2*ZCE_GaAs(n,V)*(h-ZCE_GaAs(n,V)) + w*ZCE_GaAs(n,V)
    Vcore  = (w-2*ZCE_GaAs(n,V))*(h-ZCE_GaAs(n,V))
    return Vshell*1e14,Vcore*1e14
    
    
def Brewster(l_onde,mat,x=0,y=0):
    eps = call_epsilon(l_onde,mat,x,y)
    return np.arctan(np.sqrt(eps))
    
    
def Purcell(l_onde,eps,Q,V):
    """ Retourne le facteur de Purcell, l_onde en microns, V en m^3"""
    return (3/(4*np.pi**2))*((l_onde*1e-6)/np.sqrt(eps))**3*(Q/V)
def Fermi(En,T):
    """ En en eV, T en K"""
    ev = 1.60218e-19
    kb = 1.380e-23
    return 1/(1+np.exp(En/(kb*T/ev)))
def Boltzmann(En,T):
    """ En en eV, T en K"""
    ev = 1.60218e-19
    kb = 1.380e-23
    return np.exp(-En/(kb*T/ev))
    
def Lum_CN(l_onde,T) :
    kb = 1.380e-23
    hb =1.05458e-34
    c= 3e8
    return ((c**2*hb*4*np.pi)/((l_onde*1e-6)**5))*(1/(np.exp(hb*2*np.pi*c/(kb*l_onde*1e-6*T))-1))

def Dawson_int(x,n=1e4) :
    """ Retourne l'intégrale de Dawson : D(x) = epx(-x**2)int_0^x(exp(t**2))dt. \n
    x = valeur input\n
    n = nombre de points pour calculer l'integrale"""
    # calcul de l'intégrale
    int_domain = np.linspace(0,x,n)
    integ = 0
    for val in int_domain :
        integ = integ + (np.exp(val**2)*(int_domain[1]-int_domain[0]))
    return np.exp(-x**2)*integ
    
    
def Self_energ(A,w,G,v_liste):
    g = np.zeros(v_liste.shape)
    g = np.array(g,dtype=complex)
    ii=0
    for v in v_liste :
        g[ii] = (2*A/np.sqrt(np.pi))*(Dawson_int((2*np.sqrt(np.log(2))*(v+w))/G)-Dawson_int((2*np.sqrt(np.log(2))*(v-w))/G))+ 1j*A*np.exp(-(4*np.log(2)*(v-w)**2)/G**2)  - 1j*A*np.exp(-(4*np.log(2)*(v+w)**2)/G**2)
        ii=ii+1
    return g
    
    
def call_epsilon_Al2O3(l_onde,Etude):
    """ retourne la fonction complexe anisotrope uniaxiale du sapphire : epsilon_paralell, epsilon_perp en fonction de l_onde en microns\n
    Etude : selection du papier de reference : \n
    Barker \n
    Gervais \n
    Schubert \n"""
    
    v = 1e4/l_onde
    if Etude == 'Barker':
        #Model Parameters # Barker
        wTE1 = 385
        wTE2 = 442
        wTE3 = 569
        wTE4 = 635
        
        wTA1 = 400
        wTA2 = 583
        
        wLE1 = 388
        wLE2 = 480
        wLE3 = 625
        wLE4 = 900
        
        wLA1 = 512
        wLA2 = 871
        
        gTE1 = 5.7
        gTE2 = 4.4
        gTE3 = 11.4
        gTE4 = 12.7
        
        gTA1 = 8
        gTA2 = 20.4
        
        epsinfE = 3.064
        #eps0E =  8.9
        
        epsinfA = 3.038
        #eps0A = 11.11
        
        epsilonE = epsinfE*(1+(wLE1**2-wTE1**2)/(wTE1**2-v**2-1j*gTE1*v) + (wLE2**2-wTE2**2)/(wTE2**2-v**2-1j*gTE2*v)+ (wLE3**2-wTE3**2)/(wTE3**2-v**2-1j*gTE3*v)+ (wLE4**2-wTE4**2)/(wTE4**2-v**2-1j*gTE4*v))
        epsilonA =  epsinfA*(1+(wLA1**2-wTA1**2)/(wTA1**2-v**2-1j*gTA1*v) + (wLA2**2-wTA2**2)/(wTA2**2-v**2-1j*gTA2*v))
    elif Etude == 'Gervais':
        wTE1 = 384.6
        wTE2 = 439.3
        wTE3 = 569.5
        wTE4 = 635
        
        wTA1 = 399.5
        wTA2 = 584
        
        wLE1 = 387.7
        wLE2 = 482
        wLE3 = 630.5
        wLE4 = 908
        
        wLA1 = 514
        wLA2 = 886.5
        
        gTE1 = 4.8
        gTE2 = 3.8
        gTE3 = 7.4
        gTE4 = 6.3
        
    elif Etude =='Schubert':
    
        wTE1 = 384.9
        wTE2 = 439.1
        wTE3 = 569
        wTE4 = 633.63
        
        wTA1 = 397.52
        wTA2 = 582.41
        
        wLE1 = 387.6
        wLE2 = 481.68
        wLE3 = 629.50
        wLE4 = 906.6
        
        wLA1 = 510.87
        wLA2 = 881.1
        
        gTE1 = 3.3
        gTE2 = 3.1
        gTE3 = 4.7
        gTE4 = 5
        
        gTA1 = 5.3
        gTA2 = 3
        
        gLE1 = 3.1
        gLE2 = 1.9
        gLE3 = 5.9
        gLE4 = 14.7
        
        gLA1 = 1.1
        gLA2 = 15.4
        
        epsinfE = 3.077
        #eps0E =  9.385
        
        epsinfA = 3.072 
        #eps0A = 11.614
        
        epsilonE =  epsinfE*(((wLE1**2)-(v*v)-1j*v*gLE1)*((wLE2**2)-(v*v)-1j*v*gLE2)*((wLE3**2)-(v*v)-1j*v*gLE3)*((wLE4**2)-(v*v)-1j*v*gLE4))/\
        (((wTE1**2)-(v*v)-1j*v*gTE1)*((wTE2**2)-(v*v)-1j*v*gTE2)*((wTE3**2)-(v*v)-1j*v*gTE3)*((wTE4**2)-(v*v)-1j*v*gTE4))
        epsilonA =  epsinfA*(((wLA1**2)-(v*v)-1j*v*gLA1)*((wLA2**2)-(v*v)-1j*v*gLA2))/\
        (((wTA1**2)-(v*v)-1j*v*gTA1)*((wTA2**2)-(v*v)-1j*v*gTA2))
        
    return epsilonE,epsilonA
        
def call_epsilon_AlN(l_onde):
    """ Model adapted from Moore et al. APL86 141912, for wurtzite AlN, return epsilon parallel (A) and perpendicular (E) to c-axis (z)"""
    v=1e4/l_onde
    epsinfE = 4.160
    epsinfA = 4.350
    wLE = 909.6
    wLA = 888.9
    wTE = 667.2
    wTA = 608.5
    g   = 2.2
    
    epsilonE = epsinfE*(1+(wLE**2-wTE**2)/(wTE**2-v**2-1j*g*v))
    epsilonA = epsinfA*(1+(wLA**2-wTA**2)/(wTA**2-v**2-1j*g*v))
      
    return epsilonE,epsilonA
    
def call_epsilon_GaN(l_onde):
    """ Model adapted from Kasic et al. PRB62 7365, for wurtzite GaN, return epsilon parallel (A) and perpendicular (E) to c-axis (z)"""
    v=1e4/l_onde
    epsinfE = 5.04
    epsinfA = 5.01
    wLE = 742.1
    wLA = 732.5
    wTE = 560.1
    wTA = 537
    gLE = 3.8
    gLA = 4 
    
    epsilonE = epsinfE*(1+(wLE**2-wTE**2)/(wTE**2-v**2-1j*gLE*v))
    epsilonA = epsinfA*(1+(wLA**2-wTA**2)/(wTA**2-v**2-1j*gLA*v))
      
    return epsilonE,epsilonA    

def call_epsilon_AlGaN(l_onde,x):
    """ pour le moment avec les données de Yu. Etude en incidence normale, donc uniquement modes E\n
        x = 0.05, 0.14, 0.21, 0.26, 0.32, 0.38, 0.42"""
        
    v = 1e4/l_onde
    epsinf = 3.12
    if x==0.05:
        S1  = 3.61
        wT1 = 560
        g1  = 2.3
        S2  = 0.05
        wT2 = 640
        g2  = 17.1
    elif x==  0.14:
        S1  = 3.85
        wT1 = 560
        g1  = 3.8
        S2  = 0.1
        wT2 = 648
        g2  = 18.3
    elif x==  0.21:
        S1  = 3.68
        wT1 = 570
        g1  = 7.1
        S2  = 0.52
        wT2 = 638
        g2  = 31.8
    elif x== 0.26 :
        S1  = 3.61
        wT1 = 568
        g1  = 7.4
        S2  = 0.47
        wT2 = 638
        g2  = 33.4
    elif x== 0.32 :
        S1  = 3.48
        wT1 = 574
        g1  = 15.1
        S2  = 0.83
        wT2 = 641
        g2  = 35.7
    elif x==  0.38:
        S1  = 3.25
        wT1 = 583
        g1  = 24.8
        S2  = 1.05
        wT2 = 644
        g2  = 31.2
        
    elif x== 0.42 :
        S1  = 3.21
        wT1 = 584
        g1  = 18.6
        S2  = 1.1
        wT2 = 643
        g2  = 28.8
    else :
        print('mauvaise valeur x (0.05, 0.14, 0.21, 0.26, 0.32, 0.38, 0.42)')
        S1  = 1
        wT1 = 1
        g1  = 1
        S2  = 1
        wT2 = 1
        g2  = 1
    
    epsilon = epsinf*(1+(S1*wT1**2)/(wT1**2-v**2-1j*g1*v)+(S2*wT2**2)/(wT2**2-v**2-1j*g2*v))
    return epsilon
    
    
def taucmm1_ncmm3(n):
    """ return the damping value in cm-1 for a dopage of n (cm-3), based on a simple 3degree fit of data from Kukharskii (SSC13,1761)"""
    # Données issues de Kukharaskii : Solid States Communications 13, 1761 (1973) : Plasmon-phonon coupling in GaAs
    wp_cmm1     = np.array([156, 193, 202, 357, 692, 748, 802])
    tau_cmm1    = np.array([75, 84, 86, 88, 95, 102, 105])
    wp_n        = convert_wpcmm1_dop(wp_cmm1)
    p           = np.poly1d(np.polyfit(wp_n,tau_cmm1,3))
    return p(n)
def HM_taucmm1_ncmm3(n):
    """ from my measurements... only two points for now..."""
    
    #wp          = np.array([1.5e18, 3e18])
    #tau_cmm1    = np.array([59, 70])
    #p           = np.poly1d(np.polyfit(wp,tau_cmm1,1))
    
    #return p(n)
    return 70
    
def Read_JPKsweep(Path):
    x,y,z=[],[],[]
    index_line = []
    fs = open(Path, 'r')
    #print('Open new fic') 
    #index_array = 0
    while 1: 
        txt = fs.readline()
        #print(txt)
        if ((txt =='')|(txt == '\r\n')): 
            break
        if txt[0] =='#':
            pass
        else:
            #print(txt)
            ii=-1
            index_line=[]
            while 1: # on cherche le premier espace qui limite le premier nombre
                ii = ii+1 
                if (txt[ii:ii+1] == '\t'):
                    index_line.append(ii)
                if (txt[ii:ii+4] == '\r\n'):
                    break
            x.append(float(txt[:index_line[0]]))
            y.append(float(txt[index_line[0]+1:index_line[1]]))
            z.append(float(txt[index_line[1]+1:]))  
    
            
    fs.close()
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    return x,y,z
    
    
def Read_JPK_image(Path):
    """axex,axey,image = Read_JPK_image(Path), axex and axey in mircrons"""
    
    index_line = []
    fs = open(Path, 'r')
    # I perform a quick first look at the file to get the matrix size and dimensions from the image.
    # These informations are contained in the file headers.
    # All unit are in meters
    
    for i_line in np.linspace(0,10,11):
        txt = fs.readline()
        if i_line == 5:
            print (txt[11:])
            x_size = np.float(txt[11:]) # dimension en microns fast scan
        if i_line == 6:
            print(txt[11:])
            y_size = np.float(txt[11:]) # dimension en microns slow scan
        if i_line == 9:
            x_dim = np.int(txt[10:]) # dimension en pixels fast scan
        if i_line == 10:
            y_dim = np.int(txt[10:]) # dimension en pixels slow scan
#        if i_line == 13:
#            Scan_rate = np.int(txt[10:]) # Scan rate in Hz
            
    image = np.ndarray((y_dim,x_dim), dtype = float) # Create the image matrix
    axex = np.linspace(0,x_size,x_dim)*1e6 #microns
    axey = np.linspace(0,y_size,y_dim)*1e6 #microns
    
    #print('Open new fic') 
    #index_array = 0
    nb_line = -1 # I need to count the lines to fill the matrix
    while 1: 
        txt = fs.readline()
        if ((txt =='')|(txt == '\r\n')): 
            break
        if txt[0] =='#':
            pass
        else:
            #print(txt)
            ii=-1
            index_line=[]
            while 1: # on cherche le premier espace qui limite le premier nombre
                ii = ii+1
                if (txt[ii:ii+1] == ' '):
                    index_line.append(ii)
                if (txt[ii:ii+4] == '\r\n'):
                    break

    #       ici j'ai tous mes index normalement
            
            line = []
            line.append(txt[:index_line[0]])
            index_line = np.array(index_line) # premier nombre
            for ii in range (index_line.size -1):
                line.append(np.float(txt[index_line[ii]:index_line[ii+1]]))
            # Il me manque le dernier aussi
            line.append(np.float(txt[index_line[-1]:]))
            image[nb_line,:] = line
            nb_line = nb_line+1
            #Je me suis un peu foiré là.... La première ligne est la dernière ligne... merde....
    dummy = np.zeros(image.shape)
    dummy[1:,:] = image[:-1,:]
    dummy[0,:] = image[-1,:]
    image = dummy
    fs.close()
    return axex,axey,image

def Read_JPK_image2(Path):
    """Version 2 returns also scan rate Scan_rate,axex,axey,image = Read_JPK_image(Path), axex and axey in mircrons"""
    index_line = []
    fs = open(Path, 'r')
    # I perform a quick first look at the file to get the matrix size and dimensions from the image.
    # These informations are contained in the file headers.
    # All unit are in meters
    
    for i_line in np.linspace(0,13,14):
        txt = fs.readline()
        if i_line == 5:
            print (txt[11:])
            x_size = np.float(txt[11:]) # dimension en microns fast scan
        if i_line == 6:
            print(txt[11:])
            y_size = np.float(txt[11:]) # dimension en microns slow scan
        if i_line == 9:
            x_dim = np.int(txt[10:]) # dimension en pixels fast scan
        if i_line == 10:
            y_dim = np.int(txt[10:]) # dimension en pixels slow scan
        if i_line == 13:
            Scan_rate = np.float(txt[12:]) # Scan rate in Hz
            
    image = np.ndarray((y_dim,x_dim), dtype = float) # Create the image matrix
    axex = np.linspace(0,x_size,x_dim) 
    axey = np.linspace(0,y_size,y_dim) 
    
    #print('Open new fic') 
    #index_array = 0
    nb_line = -1 # I need to count the lines to fill the matrix
    while 1: 
        txt = fs.readline()
        if ((txt =='')|(txt == '\r\n')): 
            break
        if txt[0] =='#':
            pass
        else:
            #print(txt)
            ii=-1
            index_line=[]
            while 1: # on cherche le premier espace qui limite le premier nombre
                ii = ii+1
                if (txt[ii:ii+1] == ' '):
                    index_line.append(ii)
                if (txt[ii:ii+4] == '\r\n'):
                    break

    #       ici j'ai tous mes index normalement
            
            line = []
            line.append(txt[:index_line[0]])
            index_line = np.array(index_line) # premier nombre
            for ii in range (index_line.size -1):
                line.append(np.float(txt[index_line[ii]:index_line[ii+1]]))
            # Il me manque le dernier aussi
            line.append(np.float(txt[index_line[-1]:]))
            image[nb_line,:] = line
            nb_line = nb_line+1
            #Je me suis un peu foiré là.... La première ligne est la dernière ligne... merde....
    dummy = np.zeros(image.shape)
    dummy[1:,:] = image[:-1,:]
    dummy[0,:] = image[-1,:]
    image = dummy
    fs.close()
    return Scan_rate,axex,axey,image
    
def Read_FLIM(Path):
    fs = open(Path, 'r')
    txt = fs.readline()
    x_size = np.int(txt[ 13:17])
    y_size = np.int(txt[ 19:])
    
    # ensuite des trucs pas utiles pour le moment, j'avance jusqu'a Lifetime 1
    while 1:
        txt = fs.readline()
        if txt=='Lifet. 1:\r\n':
            break
    # On commence... Lifetime 1. Chaque ligne contient x_size temps de vie, il y a y_size lignes à lire
    Lifetime1 = np.ones((x_size,y_size))
    for ii in range(y_size):
        txt = fs.readline()
        txt = txt.replace(',','.')
        for iii in range(x_size):
            Lifetime1[iii,ii] = np.float(txt[iii*24:(iii+1)*24])
            
    #F = plt.figure()
    #Fax = F.add_subplot(111)
    #Fax.pcolorfast(Lifetime)
    #F.show()           
    
    txt = fs.readline() # Amplitude 1
    
    Amp1 = np.ones((x_size,y_size))
    for ii in range(y_size):
        txt = fs.readline()
        txt = txt.replace(',','.')
        for iii in range(x_size):
            Amp1[iii,ii] = np.float(txt[iii*24:(iii+1)*24])
            
    txt = fs.readline() # Lifetime 2
    
    Lifetime2 = np.ones((x_size,y_size))
    for ii in range(y_size):
        txt = fs.readline()
        txt = txt.replace(',','.')
        for iii in range(x_size):
            Lifetime2[iii,ii] = np.float(txt[iii*24:(iii+1)*24])
                  
    txt = fs.readline() # Amplitude 2       
    
    Amp2 = np.ones((x_size,y_size))
    for ii in range(y_size):
        txt = fs.readline()
        txt = txt.replace(',','.')
        for iii in range(x_size):
            Amp2[iii,ii] = np.float(txt[iii*24:(iii+1)*24])       
              
                    
    txt = fs.readline() # Intensite  
    
    Intens = np.ones((x_size,y_size))
    for ii in range(y_size):
        txt = fs.readline()
        txt = txt.replace(',','.')
        for iii in range(x_size):
            Intens[iii,ii] = np.float(txt[iii*24:(iii+1)*24])         
    
    txt = fs.readline() # Average Lifetime  
    
    AVGLT = np.ones((x_size,y_size))
    for ii in range(y_size):
        txt = fs.readline()
        txt = txt.replace(',','.')
        for iii in range(x_size):
            AVGLT[iii,ii] = np.float(txt[iii*24:(iii+1)*24])
            
    txt = fs.readline() # bkgd
    
    bkgd = np.ones((x_size,y_size))
    for ii in range(y_size):
        txt = fs.readline()
        txt = txt.replace(',','.')
        for iii in range(x_size):
            bkgd[iii,ii] = np.float(txt[iii*24:(iii+1)*24])
    
    # Fin du fichier
    
    # Je tourne tout le monde
    Intens = np.array(zip(*Intens)[::-1])
    AVGLT = np.array(zip(*AVGLT)[::-1])
    Lifetime1 = np.array(zip(*Lifetime1)[::-1])
    Amp1 = np.array(zip(*Amp1)[::-1])
    Lifetime2 = np.array(zip(*Lifetime2)[::-1])
    Amp2 = np.array(zip(*Amp2)[::-1]) 
    
    return Intens,AVGLT,Lifetime1,Amp1,Lifetime2,Amp2
       
def Read_CCD_image(Path):
    """ axex,axey,image = st.Read_CCD_image(Path). axex and axey are in pixels."""
    fs = open(Path, 'r')
    
    #Compte le nombre de lignes, oblige pr le moment de tout lire
    # la dernière ligne est vide ! attention, j'initialise nb_line à -1 pour compenser
    nb_line = -1
    while 1: 
        txt = fs.readline()
        nb_line = nb_line+1
        if ((txt =='')|(txt == '\r\n')): 
            break
    fs.close()
    
    
    # je lis une ligne, compte le nombre d'espace et en deduit le nombre de colonne de la matrice
    fs = open(Path, 'r')
    txt = fs.readline()
    ii = 0
    index_line = []
    while 1: # on cherche le premier espace qui limite le premier nombre
        ii = ii+1 
        if (txt[ii:ii+1] == '\t'):
            index_line.append(ii)
        if (txt[ii:ii+4] == '\r\n'):
            break
    nb_col = np.array(index_line).size
    fs.close()
    
    image = np.ones((nb_line,nb_col), dtype = float) # Create the image matrix
    # Pour les axes, je reprends les chiffres obtenus lors de la calibration du mouvement de la pointe.... cad 31nm/pixel...
    #axex = np.linspace(0,0.032*nb_line,nb_line) #microns
    #axey = np.linspace(0,0.032*nb_col,nb_col) #microns
    axex = np.linspace(0,nb_line,nb_line) #pixels
    axey = np.linspace(0,nb_col,nb_col) #pixels
    
    fs = open(Path, 'r')
    
    nb_line = 0 # I need to count the lines to fill the matrix
    while 1: 
        txt = fs.readline()
        if ((txt =='')|(txt == '\r\n')): 
            break
        if txt[0] =='#':
            pass
        else:
            #print(txt)
            ii=-1
            index_line=[]
            while 1: # on cherche le premier espace qui limite le premier nombre
                ii = ii+1 
                if (txt[ii:ii+1] == '\t'):
                    index_line.append(ii)
                if (txt[ii:ii+4] == '\r\n'):
                    break
    #       ici j'ai tous mes index d'espace pour une ligne normalement
            line = []
            line.append(txt[:index_line[0]])
            index_line = np.array(index_line) # premier nombre
            for ii in range (index_line.size -1):
                line.append(np.float(txt[index_line[ii]:index_line[ii+1]]))
            # Il me manque le dernier aussi
            #line.append(np.float(txt[index_line[-1]:]))            
            image[nb_line,:] = line
            nb_line = nb_line+1
    #flipping up-down with [::-1,...] then image appears in Python as in the screen in HiPic        
    return axex,axey,image[::-1,...]
    
def Read_PicoHarp(Path):
    fs = open(Path, 'r')
    
    #first, find numbe of curves
    i_lignes = -1
    while 1: 
        txt = fs.readline()
        i_lignes = i_lignes +1
        if ((txt =='')|(txt == '\r\n')): 
            break
        
    
        if txt[:14] =="#display curve":
            txt = fs.readline() # je passe a la ligne d apres
            n_trace = 0
            ii=-1
            while 1:
                ii = ii+1
                if (txt[ii] == '\t'):
                    n_trace = n_trace+1
                if txt[ii:ii+2]=='\r\n':
                    break
        
        
        if txt[:11] =="#ns/channel":
           time_unit = np.zeros(n_trace)
           txt = fs.readline()
           iindex = []
           ii = -1
           while 1:
               ii = ii+1
               if (txt[ii] == '\t'):
                   iindex.append(ii)
               if txt[ii:ii+2]=='\r\n':
                   break
           time_unit[0]= np.float(txt[:iindex[0]].replace(',','.'))
           for i_tu in range(n_trace-1):
               time_unit[i_tu+1] = np.float(txt[iindex[i_tu]:iindex[i_tu+1]].replace(',','.'))
                 
    fs.close()
    n_trace = n_trace
    i_lignes = i_lignes -8 # il y a 10 lignes de commentaires +2 sautées pour trouver des infos...
    Mat = np.zeros((i_lignes,n_trace))
    time = np.linspace(0,i_lignes,i_lignes)
    fs = open(Path, 'r')
    ii = -1
    while 1: 
        txt = fs.readline()
        ii = ii+1
        if ((txt =='')|(txt == '\r\n')): 
            break
        if ii>=10:
            iindex=[]
            iii=-1
            while 1:
                iii = iii+1
                if (txt[iii] == '\t'):
                    iindex.append(iii)
                if txt[iii:iii+2]=='\r\n':
                    break
            Mat[ii-10,0] = txt[:iindex[0]]        
            for i_c in range(n_trace-1):
                Mat[ii-10,i_c+1] = txt[iindex[i_c]:iindex[i_c+1]]
    return time,time_unit,Mat
    
def Read_HydraHarp(Path):
    fs = open(Path, 'r')
    
    #first, find numbe of curves
    i_lignes = -1
    while 1: 
        txt = fs.readline()
        i_lignes = i_lignes +1
        if ((txt =='')|(txt == '\r\n')): 
            break
        
    
        if txt[:14] =="#display curve":
            txt = fs.readline() # je passe a la ligne d apres
            n_trace = 0
            ii=-1
            while 1:
                ii = ii+1
                if (txt[ii] == '\t'):
                    n_trace = n_trace+1
                if txt[ii:ii+2]=='\r\n':
                    break
        
        
        if txt[:7]=="#ns/bin":
           time_unit = np.zeros(n_trace)
           txt = fs.readline()
           iindex = []
           ii = -1
           while 1:
               ii = ii+1
               if (txt[ii] == '\t'):
                   iindex.append(ii)
               if txt[ii:ii+2]=='\r\n':
                   break
           time_unit[0]= np.float(txt[:iindex[0]].replace(',','.'))
           for i_tu in range(n_trace-1):
               time_unit[i_tu+1] = np.float(txt[iindex[i_tu]:iindex[i_tu+1]].replace(',','.'))
                 
    fs.close()
    n_trace = n_trace
    i_lignes = i_lignes -8 # il y a 10 lignes de commentaires +2 sautées pour trouver des infos...
    Mat = np.zeros((i_lignes,n_trace))
    time = np.linspace(0,i_lignes,i_lignes)
    fs = open(Path, 'r')
    ii = -1
    while 1: 
        txt = fs.readline()
        ii = ii+1
        if ((txt =='')|(txt == '\r\n')): 
            break
        if ii>=10:
            iindex=[]
            iii=-1
            while 1:
                iii = iii+1
                if (txt[iii] == '\t'):
                    iindex.append(iii)
                if txt[iii:iii+2]=='\r\n':
                    break
            Mat[ii-10,0] = txt[:iindex[0]]        
            for i_c in range(n_trace-1):
                Mat[ii-10,i_c+1] = txt[iindex[i_c]:iindex[i_c+1]]
    return time,time_unit,Mat
        
def Export_for_trfit(Path):
    """Read standart picoHarp txt files (copy/past in notepad) and export them as readable file for trfit module"""
    #retrieve filename and normal Path
    Filename = Path[Path.rfind('/')+1:]
    Path_save = Path[:Path.rfind('/')+1]
    time,time_unit,Mat  = Read_PicoHarp(Path)

    for ii in range(Mat.shape[1]): 
        Name = Filename[:-4] +'_'+ str(ii) +'.dat'
        fs = open(Path_save+Name,'w')
        for iii in range(Mat.shape[0]):
            fs.write(str(time[iii]*time_unit[ii]) + ' ' + str(Mat[iii,ii]) + '\n')
        
        
    #fs.write('\r\n')
    fs.close()
    print('%s files exported' %(ii))


def Calc_axe_spheroid(r,c):
    """ return the length of the short axis of a prolate spheroid of long axis radius is c, made out of a spherical nanoparticle of radius r, assuming a constant volume. """
    return np.sqrt((r**3)/c)
def Read_FLIM_1exp(Path):
    fs = open(Path, 'r')
    txt = fs.readline()
    x_size = np.int(txt[ 13:17])
    y_size = np.int(txt[ 19:])
    
    # ensuite des trucs pas utiles pour le moment, j'avance jusqu'a Lifetime 1
    while 1:
        txt = fs.readline()
        if txt=='Lifet. 1:\r\n':
            break
    # On commence... Lifetime 1. Chaque ligne contient x_size temps de vie, il y a y_size lignes à lire
    Lifetime1 = np.ones((x_size,y_size))
    for ii in range(y_size):
        txt = fs.readline()
        txt = txt.replace(',','.')
        for iii in range(x_size):
            Lifetime1[iii,ii] = np.float(txt[iii*24:(iii+1)*24])
            
    #F = plt.figure()
    #Fax = F.add_subplot(111)
    #Fax.pcolorfast(Lifetime)
    #F.show()           
    
    txt = fs.readline() # Amplitude 1
    
    Amp1 = np.ones((x_size,y_size))
    for ii in range(y_size):
        txt = fs.readline()
        txt = txt.replace(',','.')
        for iii in range(x_size):
            Amp1[iii,ii] = np.float(txt[iii*24:(iii+1)*24])
                      
    txt = fs.readline() # Intensite  
    
    Intens = np.ones((x_size,y_size))
    for ii in range(y_size):
        txt = fs.readline()
        txt = txt.replace(',','.')
        for iii in range(x_size):
            Intens[iii,ii] = np.float(txt[iii*24:(iii+1)*24])         
    
    txt = fs.readline() # Average Lifetime  
    
    AVGLT = np.ones((x_size,y_size))
    for ii in range(y_size):
        txt = fs.readline()
        txt = txt.replace(',','.')
        for iii in range(x_size):
            AVGLT[iii,ii] = np.float(txt[iii*24:(iii+1)*24])
            
    txt = fs.readline() # bkgd
    
    bkgd = np.ones((x_size,y_size))
    for ii in range(y_size):
        txt = fs.readline()
        txt = txt.replace(',','.')
        for iii in range(x_size):
            bkgd[iii,ii] = np.float(txt[iii*24:(iii+1)*24])
    
    # Fin du fichier
    
    # Je tourne tout le monde
    Intens = np.rot90(np.fliplr(np.flipud(Intens)))
    AVGLT = np.rot90(np.fliplr(np.flipud(AVGLT)))
    Lifetime1 = np.rot90(np.fliplr(np.flipud(Lifetime1)))
    Amp1 = np.rot90(np.fliplr(np.flipud(Amp1)))
    
    return Intens,AVGLT,Lifetime1,Amp1

def Read_FLIM_Ionly(Path):
    fs = open(Path, 'r')
    txt = fs.readline()
    x_size = np.int(txt[ 13:17])
    y_size = np.int(txt[ 19:])
    
    # ensuite des trucs pas utiles pour le moment, j'avance jusqu'a Lifetime 1
    while 1:
        txt = fs.readline()
        if txt=='Intens. :\r\n':
            break
    
    Intens = np.ones((x_size,y_size))
    for ii in range(y_size):
        txt = fs.readline()
        txt = txt.replace(',','.')
        for iii in range(x_size):
            Intens[iii,ii] = np.float(txt[iii*24:(iii+1)*24])         
    
    # Je tourne tout le monde
    #Intens = np.array(zip(*Intens)[::-1])
    Intens = np.rot90(np.fliplr(np.flipud(Intens)))
    #Intens = np.fliplr(Intens)
    #Intens = np.flipud(Intens)
    fs.close()
    
    return Intens


def Read_PT3(Path):
    """ sync,pt,marker_frame,marker_linestart,marker_linestop,Resolution,CntRate0  = Read_PT3(Path) \n \
    Reads pt3 files. return the sync time (overall macrotime), pt time (micro time : photon arrival), \
    markers for frame, line start and line stop, Resolution and CntRate0 (rate of sync signal) and cnt_Err \n \
    For simplicity here I remove all comments, the original file used for test is 20140303_Read_bin_Symphotime, everything is detailed there plus some postprocessing \n \
    """
    f = open(Path, "rb")
    f.read(584) # Reads all the shit above
    Resolution      = struct.unpack('f',f.read(4))[0] # in ns

    f.read(116)
    CntRate0    = struct.unpack('i',f.read(4))[0]
    f.read(12)
    Records     = struct.unpack('i',f.read(4))[0]
    Hdrsize     = struct.unpack('i',f.read(4))[0] #Size of special header
    if (Hdrsize != 0): #depending of point or image mode, header is there or not... 
        ImgHdr      = struct.unpack('36i',f.read(Hdrsize*4)) 
    else :
        pass
    #f.read(148)
    ofltime     = 0
    cnt_Ofl     = 0
    cnt_lstart  = 0
    cnt_lstop   = 0
    cnt_Err     = 0
    cnt_f       = 0 #added Simon, frame counter
    WRAPAROUND  = 65536
    syncperiod  = 1e9/np.float(CntRate0) # in ns, CntRate0 is in Hz
    marker_frame,marker_linestart, marker_linestop = [],[],[] # Record marker events corresponding to line start and line stop
    pt          = np.zeros(np.int(Records))
    sync        = np.zeros(np.int(Records))
    for ii in range(np.int(Records)):
        T3Record    = struct.unpack('I',f.read(4))
        A           = T3Record[0]
        chan        = (A&((2**4-1)<<(32-4)))>>(32-4) # Read highest 4 bits
        nsync       = A&(2**16-1) #Read lowest 16 bits
        dtime       = 0
        if ((chan ==1)|(chan ==2)|(chan ==3)|(chan ==4)):
            dtime       = (A&(2**12-1)<<16)>>16
        elif chan == 15:
            markers     = (A&(2**4-1)<<16)>>16
            # Then depending on the marker value there are different possibilties
            if markers == 0 :#This is a time overflow
                ofltime     = ofltime + WRAPAROUND
                cnt_Ofl     = cnt_Ofl+1
            elif markers == 1: # it is a true marker, 1 is for Frame start/stop
                #print (markers)
                cnt_f       = cnt_f+1
                marker_frame.append((ofltime + nsync)*syncperiod)
            elif markers == 2 :# Here I take all the other markers without any differentiation (2 in linestart, 4 is linestop)
                #print (markers)
                cnt_lstart  = cnt_lstart+1
                marker_linestart.append((ofltime + nsync)*syncperiod)
            elif markers == 4:
                #print (markers)
                cnt_lstop   = cnt_lstop+1
                marker_linestop.append((ofltime + nsync)*syncperiod)
        else : # There is an error, should not happen in T3 Mode (says the PicoQuant Matlab code)... Still got a lot...
            cnt_Err     = cnt_Err+1
        
        truensync       = ofltime+nsync
        truetime        = truensync*syncperiod + dtime*Resolution
        pt[ii]          = truetime
        sync[ii]        = truensync*syncperiod

    marker_frame        = np.array(marker_frame)
    marker_linestart    = np.array(marker_linestart)
    marker_linestop     = np.array(marker_linestop)
    f.close()
    return sync,pt,marker_frame,marker_linestart,marker_linestop,Resolution,CntRate0,cnt_Err 

def Read_PT3_b(Path):
    """ sync,pt,marker_frame,marker_linestart,marker_linestop,Resolution,CntRate0  = Read_PT3(Path) \n \
    Reads pt3 files. return the sync time (overall macrotime), pt time (micro time : photon arrival), \
    markers for frame, line start and line stop, Resolution and CntRate0 (rate of sync signal) and cnt_Err \n \
    For simplicity here I remove all comments, the original file used for test is 20140303_Read_bin_Symphotime, everything is detailed there plus some postprocessing \n \
    """
    Hist        = np.zeros(4096)
    f           = open(Path, "rb")
    f.read(584) # Reads all the shit above
    Resolution  = struct.unpack('f',f.read(4))[0] # in ns
    Time        = np.linspace(0,4095*Resolution,4096) # see later, dtime is coded on 12 bits (max = (2**11)-1)
    f.read(116)
    CntRate0    = struct.unpack('i',f.read(4))[0]
    f.read(12)
    Records     = struct.unpack('i',f.read(4))[0]
    Hdrsize     = struct.unpack('i',f.read(4))[0] #Size of special header
    if (Hdrsize != 0): #depending of point or image mode, header is there or not... 
        ImgHdr      = struct.unpack('36i',f.read(Hdrsize*4)) 
    else :
        pass
    #f.read(148)
    ofltime     = 0
    cnt_Ofl     = 0
    cnt_lstart  = 0
    cnt_lstop   = 0
    cnt_Err     = 0
    cnt_f       = 0 #added Simon, frame counter, doesn't seem to work...
    WRAPAROUND  = 65536
    syncperiod  = 1e9/np.float(CntRate0) # in ns, CntRate0 is in Hz
    marker_frame,marker_linestart, marker_linestop = [],[],[] # Record marker events corresponding to line start and line stop
    pt          = np.zeros(np.int(Records))
    sync        = np.zeros(np.int(Records))
    for ii in range(np.int(Records)):
        T3Record    = struct.unpack('I',f.read(4))
        A           = T3Record[0]
        chan        = (A&((2**4-1)<<(32-4)))>>(32-4) # Read highest 4 bits
        nsync       = A&(2**16-1) #Read lowest 16 bits
        dtime       = 0
        if ((chan ==1)|(chan ==2)|(chan ==3)|(chan ==4)):
            dtime       = (A&(2**12-1)<<16)>>16
            Hist[dtime] = Hist[dtime]+1
            
        elif chan == 15:
            markers     = (A&(2**4-1)<<16)>>16
            # Then depending on the marker value there are different possibilties
            if markers == 0 :#This is a time overflow
                ofltime         = ofltime + WRAPAROUND
                cnt_Ofl         = cnt_Ofl+1
                #truensync       = ofltime+nsync
                #truetime        = #truensync*syncperiod + dtime*Resolution
                #pt[ii]          = dtime*Resolution
                #sync[ii]        = truensync*syncperiod
            elif markers == 1: # it is a true marker, 1 is for Frame start/stop
                #print (markers)
                cnt_f       = cnt_f+1
                marker_frame.append((ofltime + nsync)*syncperiod)
            elif markers == 2 :# Here I take all the other markers without any differentiation (2 in linestart, 4 is linestop)
                #print (markers)
                cnt_lstart  = cnt_lstart+1
                marker_linestart.append((ofltime + nsync)*syncperiod)
            elif markers == 4:
                #print (markers)
                cnt_lstop   = cnt_lstop+1
                marker_linestop.append((ofltime + nsync)*syncperiod)
            else : # There is an error, should not happen in T3 Mode (says the PicoQuant Matlab code)... Still got a lot...
                cnt_Err     = cnt_Err+1
        truensync       = ofltime+nsync
        #truetime        = truensync*syncperiod + dtime*Resolution
        pt[ii]          = dtime*Resolution
        sync[ii]        = truensync*syncperiod 

    marker_frame        = np.array(marker_frame)
    marker_linestart    = np.array(marker_linestart)
    marker_linestop     = np.array(marker_linestop)
    f.close()
    return Time,Hist,sync,pt,marker_frame,marker_linestart,marker_linestop,Resolution,CntRate0,cnt_Err 


def read_spe(spefilename, verbose=False):
    """ 
    Read a binary PI SPE file into a python dictionary

    Inputs:

        spefilename --  string specifying the name of the SPE file to be read
        verbose     --  boolean print debug statements (True) or not (False)

        Outputs
        spedict     
        
            python dictionary containing header and data information
            from the SPE file
            Content of the dictionary is:
            spedict = {'data':[],    # a list of 2D numpy arrays, one per image
            'IGAIN':pimaxGain,
            'EXPOSURE':exp_sec,
            'SPEFNAME':spefilename,
            'OBSDATE':date,
            'CHIPTEMP':detectorTemperature
            }

    I use the struct module to unpack the binary SPE data.
    Some useful formats for struct.unpack_from() include:
    fmt   c type          python
    c     char            string of length 1
    s     char[]          string (Ns is a string N characters long)
    h     short           integer 
    H     unsigned short  integer
    l     long            integer
    f     float           float
    d     double          float

    The SPE file defines new c types including:
        BYTE  = unsigned char
        WORD  = unsigned short
        DWORD = unsigned long


    Example usage:
    Given an SPE file named test.SPE, you can read the SPE data into
    a python dictionary named spedict with the following:
    >>> import piUtils
    >>> spedict = piUtils.readSpe('test.SPE')
    """
  
    # open SPE file as binary input
    spe = open(spefilename, "rb")
    
    # Header length is a fixed number
    nBytesInHeader = 4100

    # Read the entire header
    header = spe.read(nBytesInHeader)
    
    # version of WinView used
    swversion = struct.unpack_from("16s", header, offset=688)[0]
    
    # version of header used
    # Eventually, need to adjust the header unpacking
    # based on the headerVersion.  
    headerVersion = struct.unpack_from("f", header, offset=1992)[0]
  
    # which camera controller was used?
    controllerVersion = struct.unpack_from("h", header, offset=0)[0]
    if verbose:
        print ("swversion         = ", swversion)
        print ("headerVersion     = ", headerVersion)
        print ("controllerVersion = ", controllerVersion)
    
    # Date of the observation
    # (format is DDMONYYYY  e.g. 27Jan2009)
    date = struct.unpack_from("9s", header, offset=20)[0]
    
    # Exposure time (float)
    exp_sec = struct.unpack_from("f", header, offset=10)[0]
    
    # Intensifier gain
    pimaxGain = struct.unpack_from("h", header, offset=148)[0]

    # Not sure which "gain" this is
    gain = struct.unpack_from("H", header, offset=198)[0]
    
    # Data type (0=float, 1=long integer, 2=integer, 3=unsigned int)
    data_type = struct.unpack_from("h", header, offset=108)[0]

    comments = struct.unpack_from("400s", header, offset=200)[0]

    # CCD Chip Temperature (Degrees C)
    detectorTemperature = struct.unpack_from("f", header, offset=36)[0]

    # The following get read but are not used
    # (this part is only lightly tested...)
    analogGain = struct.unpack_from("h", header, offset=4092)[0]
    noscan = struct.unpack_from("h", header, offset=34)[0]
    pimaxUsed = struct.unpack_from("h", header, offset=144)[0]
    pimaxMode = struct.unpack_from("h", header, offset=146)[0]

    ########### here's from Kasey
    #int avgexp 2 number of accumulations per scan (why don't they call this "accumulations"?)
    #TODO: this isn't actually accumulations, so fix it...    
    accumulations = struct.unpack_from("h", header, offset=668)[0]
    if accumulations == -1:
        # if > 32767, set to -1 and 
        # see lavgexp below (668) 
        #accumulations = struct.unpack_from("l", header, offset=668)[0]
        # or should it be DWORD, NumExpAccums (1422): Number of Time experiment accumulated        
        accumulations = struct.unpack_from("l", header, offset=1422)[0]
        
    """Start of X Calibration Structure (although I added things to it that I thought were relevant,
       like the center wavelength..."""
    xcalib = {}
    
    #SHORT SpecAutoSpectroMode 70 T/F Spectrograph Used
    xcalib['SpecAutoSpectroMode'] = bool( struct.unpack_from("h", header, offset=70)[0] )

    #float SpecCenterWlNm # 72 Center Wavelength in Nm
    xcalib['SpecCenterWlNm'] = struct.unpack_from("f", header, offset=72)[0]
    
    #SHORT SpecGlueFlag 76 T/F File is Glued
    xcalib['SpecGlueFlag'] = bool( struct.unpack_from("h", header, offset=76)[0] )

    #float SpecGlueStartWlNm 78 Starting Wavelength in Nm
    xcalib['SpecGlueStartWlNm'] = struct.unpack_from("f", header, offset=78)[0]

    #float SpecGlueEndWlNm 82 Starting Wavelength in Nm
    xcalib['SpecGlueEndWlNm'] = struct.unpack_from("f", header, offset=82)[0]

    #float SpecGlueMinOvrlpNm 86 Minimum Overlap in Nm
    xcalib['SpecGlueMinOvrlpNm'] = struct.unpack_from("f", header, offset=86)[0]

    #float SpecGlueFinalResNm 90 Final Resolution in Nm
    xcalib['SpecGlueFinalResNm'] = struct.unpack_from("f", header, offset=90)[0]

    #  short   BackGrndApplied              150  1 if background subtraction done
    xcalib['BackgroundApplied'] = struct.unpack_from("h", header, offset=150)[0]
    BackgroundApplied=False
    if xcalib['BackgroundApplied']==1: BackgroundApplied=True

    #  float   SpecGrooves                  650  Spectrograph Grating Grooves
    xcalib['SpecGrooves'] = struct.unpack_from("f", header, offset=650)[0]

    #  short   flatFieldApplied             706  1 if flat field was applied.
    xcalib['flatFieldApplied'] = struct.unpack_from("h", header, offset=706)[0]
    flatFieldApplied=False
    if xcalib['flatFieldApplied']==1: flatFieldApplied=True
    
    #double offset # 3000 offset for absolute data scaling */
    xcalib['offset'] = struct.unpack_from("d", header, offset=3000)[0]

    #double factor # 3008 factor for absolute data scaling */
    xcalib['factor'] = struct.unpack_from("d", header, offset=3008)[0]
    
    #char current_unit # 3016 selected scaling unit */
    xcalib['current_unit'] = struct.unpack_from("c", header, offset=3016)[0]

    #char reserved1 # 3017 reserved */
    xcalib['reserved1'] = struct.unpack_from("c", header, offset=3017)[0]

    #char string[40] # 3018 special string for scaling */
    xcalib['string'] = struct.unpack_from("40c", header, offset=3018)
    
    #char reserved2[40] # 3058 reserved */
    xcalib['reserved2'] = struct.unpack_from("40c", header, offset=3058)

    #char calib_valid # 3098 flag if calibration is valid */
    xcalib['calib_valid'] = struct.unpack_from("c", header, offset=3098)[0]

    #char input_unit # 3099 current input units for */
    xcalib['input_unit'] = struct.unpack_from("c", header, offset=3099)[0]
    """/* "calib_value" */"""

    #char polynom_unit # 3100 linear UNIT and used */
    xcalib['polynom_unit'] = struct.unpack_from("c", header, offset=3100)[0]
    """/* in the "polynom_coeff" */"""

    #char polynom_order # 3101 ORDER of calibration POLYNOM */
    xcalib['polynom_order'] = struct.unpack_from("c", header, offset=3101)[0]

    #char calib_count # 3102 valid calibration data pairs */
    xcalib['calib_count'] = struct.unpack_from("c", header, offset=3102)[0]

    #double pixel_position[10];/* 3103 pixel pos. of calibration data */
    xcalib['pixel_position'] = struct.unpack_from("10d", header, offset=3103)

    #double calib_value[10] # 3183 calibration VALUE at above pos */
    xcalib['calib_value'] = struct.unpack_from("10d", header, offset=3183)

    #double polynom_coeff[6] # 3263 polynom COEFFICIENTS */
    xcalib['polynom_coeff'] = struct.unpack_from("6d", header, offset=3263)

    #double laser_position # 3311 laser wavenumber for relativ WN */
    xcalib['laser_position'] = struct.unpack_from("d", header, offset=3311)[0]

    #char reserved3 # 3319 reserved */
    xcalib['reserved3'] = struct.unpack_from("c", header, offset=3319)[0]

    #unsigned char new_calib_flag # 3320 If set to 200, valid label below */
    #xcalib['calib_value'] = struct.unpack_from("BYTE", header, offset=3320)[0] # how to do this?

    #char calib_label[81] # 3321 Calibration label (NULL term'd) */
    xcalib['calib_label'] = struct.unpack_from("81c", header, offset=3321)

    #char expansion[87] # 3402 Calibration Expansion area */
    xcalib['expansion'] = struct.unpack_from("87c", header, offset=3402)
    ########### end of Kasey's addition

    if verbose:
        print ("date      = ["+date+"]")
        print ("exp_sec   = ", exp_sec)
        print ("pimaxGain = ", pimaxGain)
        print ("gain (?)  = ", gain)
        print ("data_type = ", data_type)
        print ("comments  = ["+comments+"]")
        print ("analogGain = ", analogGain)
        print ("noscan = ", noscan)
        print ("detectorTemperature [C] = ", detectorTemperature)
        print ("pimaxUsed = ", pimaxUsed)

    # Determine the data type format string for
    # upcoming struct.unpack_from() calls
    if data_type == 0:
        # float (4 bytes)
        dataTypeStr = "f"  #untested
        bytesPerPixel = 4
        dtype = "float32"
    elif data_type == 1:
        # long (4 bytes)
        dataTypeStr = "l"  #untested
        bytesPerPixel = 4
        dtype = "int32"
    elif data_type == 2:
        # short (2 bytes)
        dataTypeStr = "h"  #untested
        bytesPerPixel = 2
        dtype = "int32"
    elif data_type == 3:  
        # unsigned short (2 bytes)
        dataTypeStr = "H"  # 16 bits in python on intel mac
        bytesPerPixel = 2
        dtype = "int32"  # for numpy.array().
        # other options include:
        # IntN, UintN, where N = 8,16,32 or 64
        # and Float32, Float64, Complex64, Complex128
        # but need to verify that pyfits._ImageBaseHDU.ImgCode cna handle it
        # right now, ImgCode must be float32, float64, int16, int32, int64 or uint8
    else:
        print ("unknown data type")
        print ("returning...")
        sys.exit()
  
    # Number of pixels on x-axis and y-axis
    nx = struct.unpack_from("H", header, offset=42)[0]
    ny = struct.unpack_from("H", header, offset=656)[0]
    
    # Number of image frames in this SPE file
    nframes = struct.unpack_from("l", header, offset=1446)[0]

    if verbose:
        print ("nx, ny, nframes = ", nx, ", ", ny, ", ", nframes)
    
    npixels = nx*ny
    npixStr = str(npixels)
    fmtStr  = npixStr+dataTypeStr
    if verbose:
        print ("fmtStr = ", fmtStr)
    
    # How many bytes per image?
    nbytesPerFrame = npixels*bytesPerPixel
    if verbose:
        print ("nbytesPerFrame = ", nbytesPerFrame)

    # Create a dictionary that holds some header information
    # and contains a placeholder for the image data
    spedict = {'data':[],    # can have more than one image frame per SPE file
                'IGAIN':pimaxGain,
                'EXPOSURE':exp_sec,
                'SPEFNAME':spefilename,
                'OBSDATE':date,
                'CHIPTEMP':detectorTemperature,
                'COMMENTS':comments,
                'XCALIB':xcalib,
                'ACCUMULATIONS':accumulations,
                'FLATFIELD':flatFieldApplied,
                'BACKGROUND':BackgroundApplied
                }
    
    # Now read in the image data
    # Loop over each image frame in the image
    if verbose:
        print ("Reading image frames number "),
    for ii in range(nframes):
        iistr = str(ii)
        data = spe.read(nbytesPerFrame)
        if verbose:
            print (iistr," ",)
    
        # read pixel values into a 1-D numpy array. the "=" forces it to use
        # standard python datatype size (4bytes for 'l') rather than native
        # (which on 64bit is 8bytes for 'l', for example).
        # See http://docs.python.org/library/struct.html
        dataArr = np.array(struct.unpack_from("="+fmtStr, data, offset=0),
                            dtype=dtype)

        # Resize array to nx by ny pixels
        # notice order... (y,x)
        dataArr.resize((ny, nx))
        #print dataArr.shape

        # Push this image frame data onto the end of the list of images
        # but first cast the datatype to float (if it's not already)
        # this isn't necessary, but shouldn't hurt and could save me
        # from doing integer math when i really meant floating-point...
        spedict['data'].append( dataArr.astype(float) )

    if verbose:
        print ("")
  
    return spedict



def Read_AFM_Brazil(Filename):
    """ Read the file (complete path) and return a matrix"""
    fid         = open(Filename)
    nb_lines    = 0
    while 1:    
        line = fid.readline()
        if line == '':
            break
        if nb_lines==0:
            nb_col = line.count('E')
        nb_lines = nb_lines+1
    fid.close()

    # It seems the format is always the same
    Im = np.ndarray([nb_lines,nb_col])

    fid = open(Filename)
    nb_lines = 0
    while 1:    
        line = fid.readline()
        if line == '':
            break
        for ii in range(nb_col):
            if ii ==0:
                Im[nb_lines,ii] = np.float(line[:line.find(' ')])
                tmp_line = line[line.find(' ')+1:]
            else:
                Im[nb_lines,ii] = np.float(tmp_line[:tmp_line.find(' ')])
                tmp_line = tmp_line[tmp_line.find(' ')+1:]
        nb_lines = nb_lines+1
    fid.close()
    return Im

def Read_Nanonis_sxm(Filename):
    fid         = open(Filename, 'rb')
    line        = fid.readline() #':NANONIS_VERSION:\n'
    line        = np.float(fid.readline()) #'2\n'
    line        = fid.readline() #':SCANIT_TYPE:\n'
    line        = fid.readline() #'              FLOAT            MSBFIRST\n'
    line        = fid.readline() #':REC_DATE:\n'
    Date        = fid.readline() #' 28.07.2016\n'
    line        = fid.readline() #':REC_TIME:
    Time        = fid.readline() # 15:34:20
    line        = fid.readline() #':REC_TEMP:
    line        = fid.readline() #'      290.0000000000\n'
    line        = fid.readline() #':ACQ_TIME:\n'
    Acq_Time    = np.float(fid.readline()) #'       643.5\n'
    line        = fid.readline() #':SCAN_PIXELS:\n'
    xy_size     = fid.readline().split('       ')[1:3] #'       128       128\n'
    nx          = np.float(xy_size[0])
    ny          = np.float(xy_size[1])
    line        = fid.readline() #':SCAN_FILE:\n'
    line        = fid.readline() #'C:\\Users\\admin-local\\Documents\\NANONIS_DATA\\2016\\2016-07-27\\unnamed002.sxm\n'
    line        = fid.readline() #':SCAN_TIME:\n' ?? not sure
    line        = fid.readline() #'             2.500E+0             2.500E+0\n'
    line        = fid.readline() #':SCAN_RANGE:\n'
    xy_size     = fid.readline().split('       ')[1:3] #'           5.000000E-6           5.000000E-6\n'
    xsize_meter = np.float(xy_size[0])
    ysize_meter = np.float(xy_size[1])
    line        = fid.readline() #':SCAN_OFFSET:\n'
    xy_offset   = fid.readline().split('       ')[1:3]
    x_offset    = np.float(xy_offset[0])
    y_offset    = np.float(xy_offset[1])
    line        = fid.readline() #:SCAN_ANGLE:
    scan_angle  = np.float(fid.readline()) #'            0.000E+0\n'
    line        = fid.readline() #':SCAN_DIR:\n'
    scan_dir    = fid.readline() #'up\n'
    line        = fid.readline() #':BIAS:\n'
    line        = fid.readline() #'            0.000E+0\n'
    line        = fid.readline() #':Z-CONTROLLER:\n'
    line        = fid.readline() #'\tName\ton\tSetpoint\tP-gain\tI-gain\tT-const\n'
    Zcontrol_columns = line.split('\t')[1:]
    line        = fid.readline() #'\tAmplitude\t1\t5.662E-5 m\t5.000E-4 m/m\t7.000E-2 m/m/s\t7.143E-3 s\n'
    Zcontrol_values = line.split('\t')[1:]
    line        = fid.readline() #':COMMENT:\n'
    line        = fid.readline() #'\n'
    line        = fid.readline() #':NanonisMain>Session Path:\n'
    line        = fid.readline() #'C:\\Users\\admin-local\\Documents\\NANONIS_DATA\\2016\\2016-07-27\n'
    line        = fid.readline() #':NanonisMain>SW Version:\n'
    line        = fid.readline() #'Generic 4\n'
    line        = fid.readline() #':NanonisMain>UI Release:\n'
    line        = fid.readline() #'4919\n'
    line        = fid.readline() #':NanonisMain>RT Release:\n'
    line        = fid.readline() #'4437\n'
    line        = fid.readline() #':NanonisMain>RT Frequency (Hz):\n'
    line        = fid.readline() #'15E+3\n'
    line        = fid.readline() #':NanonisMain>Signals Oversampling:\n'
    line        = fid.readline() #'10\n'
    line        = fid.readline() #':NanonisMain>Animations Period (s):\n'
    line        = fid.readline() #'20E-3\n'
    line        = fid.readline() #':NanonisMain>Indicators Period (s):\n'
    line        = fid.readline() #'300E-3\n'
    line        = fid.readline() #':NanonisMain>Measurements Period (s):\n'
    line        = fid.readline() #'16.6257E-3\n'
    line        = fid.readline() #':DATA_INFO:\n'
    #From here there is an unknown number of channels that are recorded
    line        = fid.readline() #'\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n'
    chan_info   = []
    tag         = '\n'
    while 1:
        entry = fid.readline() # gives the line content
        if  entry==tag: #check if tag is contained in entry (in the line)
            break #and then break. Carreful with general file handling: if tag is not in the files it loops forever :)
        else:
            chan_info.append(entry)
    
    line        = fid.readline() #':SCANIT_END:\n' end of header
    byteoffset  = fid.tell()
    #data begins with 4 byte code, add 4 bytes to offset instead
    byteoffset += 4
    fid.close()
    
    #Parsing header info into a dictionnary
    nchanns         = len(chan_info)
    channs          = []
    chann_dir       = []
    chann_unit      = []
    chann_calib     = []
    chann_offset    = []
    for ii in range(len(chan_info)):
        channs.append(chan_info[ii].split('\t')[2])
        chann_unit.append(chan_info[ii].split('\t')[3])
        chann_dir.append(chan_info[ii].split('\t')[4])
        chann_calib.append(np.float(chan_info[ii].split('\t')[5]))
        chann_offset.append(np.float(chan_info[ii].split('\t')[6]))
    
    
     # assume both directions for now
    ndir        = 2
    data_dict   = dict()
    f           = open(Filename, 'rb')
    f.seek(byteoffset)
    data_format = '>f4'
    scandata    = np.fromfile(f, dtype=data_format)
    f.close()
    scandata_shaped = scandata.reshape(nchanns, ndir, nx, ny)
    # extract data for each channel
    for i, chann in enumerate(channs):
        chann_dict = dict(forward=scandata_shaped[i, 0, :, :],
                          backward=scandata_shaped[i, 1, :, :])
        data_dict[chann] = chann_dict
    
    data_dict['Channel Name']       = channs
    data_dict['Channel Units']      = chann_unit
    data_dict['Channel Calib']      = chann_calib
    data_dict['Channel Offset']     = chann_offset
    data_dict['Channel dir']        = chann_dir
    data_dict['Acq Time']           = Acq_Time
    data_dict['Zdata label']        = Zcontrol_columns
    data_dict['Zdata values']       = Zcontrol_values
    data_dict['axeX']               = np.linspace(0,xsize_meter*1e6,nx)
    data_dict['axeY']               = np.linspace(0,ysize_meter*1e6,ny)
    data_dict['Xoofset, Yoffset']   = [x_offset,y_offset]
    data_dict['scan_angle']         = scan_angle
    data_dict['scan_dir']           = scan_dir
    data_dict['Date']               = Date
    data_dict['Time']               = Time
    return data_dict

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.
    Marcos Duarte, https://github.com/demotu/BMC
    version 1.0.4
    License MIT
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    
    The function can handle NaN's 

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def Correct_Comsmic_Rays(Spec):
	diff_spec = np.diff(Spec)
	Cray_pos = np.where(diff_spec>= np.mean(3*np.abs(diff_spec)))

	corr_spec = Spec.copy()

	for ii in Cray_pos:
		#Seems these rays can be 4 points wide !
		try :
			corr_spec[ii] = (corr_spec[ii-1] + corr_spec[ii+5])/2
			corr_spec[ii+1] = (corr_spec[ii] + corr_spec[ii+5])/2
			corr_spec[ii+2] = (corr_spec[ii+1] + corr_spec[ii+5])/2
			corr_spec[ii+3] = (corr_spec[ii+2] + corr_spec[ii+5])/2
		except:
			pass
	return corr_spec