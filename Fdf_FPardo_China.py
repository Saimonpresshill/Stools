# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
#import sqrts as sqrts

"""
2014-07-17 (FP) added absolute_square
2014-07-17 (FP) Fdf -> fdf in comments
2014-07-18 (FP) removed "#" lines
"""

class Fdf:

    """
from scikits import fdf
from  scikits.fdf import Fdf
import numpy as np

v = np.array([1,2,3])
v = np.array([1,2,3j])
a = fdf.variable(v)

print(a)
print(a+a)
print(a+a-2*a)
print(a*a/a - a)
print(fdf.sqrt(a**2) - a)
print(Fdf.sqrt(a**2) - a)
print((a**2).sqrt() - a)
print(fdf.square(a) - a**2)
print(fdf.square(fdf.sin(a)) + fdf.square(fdf.cos(a)))
print(fdf.exp(fdf.log(a)) - 1)
s,c = fdf.sincos(a) ; print(s**2 + c**2 - 1)


# newton:
s = np.array([1,2,3,1j,-2])
r = np.array([1,1,1,1,1j])
print(r)
for i in xrange(10):
    r = fdf.newton_step(lambda z: fdf.variable(z)**2 - fdf.constant(s), r)
    print(np.max(np.abs(np.sqrt(s)-r)))

s = np.array([1,2,3,1j,-2])
r = np.array([1,1,1,1,1j])
(r,err) = fdf.newton_ten_steps(lambda z: fdf.variable(z)**2 - fdf.constant(s), r)
print (r,err)

"""

    def __init__(self,f,df):
        """Construct a Fdf from the value of a function and the value of its derivative"""
        self.f,self.df = f,df

    def __str__(self):
        return '(' + self.f.__str__() + ', ' + self.df.__str__() + ')'

    def __repr__(self):
        return 'Fdf(' + self.f.__repr__() + ', ' + self.df.__repr__() + ')'

    def __add__(self,a):
        if isinstance(a,Fdf):
            return Fdf(self.f + a.f, self.df + a.df)
        else:
            return Fdf(self.f + a, self.df)

    def __radd__(self,a):
        if isinstance(a,Fdf):
            return Fdf(a.f + self.f, a.df + self.df)
        else:
            return Fdf(a + self.f, self.df)

    def __sub__(self,a):
        if isinstance(a,Fdf):
            return Fdf(self.f - a.f, self.df - a.df)
        else:
            return Fdf(self.f - a, self.df)

    def __rsub__(self,a):
        if isinstance(a,Fdf):
            return Fdf(a.f - self.f, a.df - self.df)
        else:
            return Fdf(a - self.f, -self.df)

    def __neg__(self):
        return Fdf(-self.f, -self.df)

    def __mul__(self,a):
        if isinstance(a,Fdf):
            return Fdf(self.f * a.f, self.f * a.df + self.df * a.f)
        else:
            return Fdf(self.f * a, self.df * a)            

    def __rmul__(self,a):
        if isinstance(a,Fdf):
            return Fdf(a.f * self.f, a.df * self.f + a.f * self.df)
        else:
            return Fdf(a * self.f, a * self.df)            

    def inv(self):
        return Fdf(1.0 / self.f, - self.df / (self.f**2))

    def __truediv__(self,a):
        if isinstance(a,Fdf):
            return self * Fdf.inv(a)
        else:
            return self * (1.0 / a)

    __div__ = __truediv__

    def __rtruediv__(self,a):
        return a * self.inv()

    __rdiv__ = __rtruediv__

    def __pow__(self,a):
        if isinstance(a,Fdf):
            spa = self.f**a.f
            return Fdf(spa, (a.df * np.log(self.f) + a.f * self.df / self.f) * spa)
        else :
            spa = self.f**a
            return Fdf(spa, (a * self.df / self.f) * spa)

    def __rpow__(self,a):
        if isinstance(a,Fdf):
            aps = a.f**self.f
            return Fdf(aps, (self.df * np.log(a.f) + self.f * a.df / a.f) * aps)
        else :
            aps = a**self.f
            return Fdf(aps, (self.df * np.log(a)) * aps)

    def __getitem__(self,i):
        return Fdf(self.f.__getitem__(i),self.df.__getitem__(i))

def variable(x):
    """Constructs a Fdf variable from its value. The derivative is 1.
    Warns, it constructs z variable having value value, not variable value*z 
    """
    if not np.isscalar(x):
        x = np.asarray(x)
    return Fdf(x,np.ones_like(x))


def constant(x):
    """Constructs a Fdf constant from its value. The derivative is 0"""
    if not np.isscalar(x):
        x = np.asarray(x)
    return Fdf(x,np.zeros_like(x))

def sqrt(v, funsqrt = np.sqrt):
    s = funsqrt(v.f)
    #print(s)
    # original :
    return Fdf(s,0.5*v.df/s)
    # This gives a problem when s comes close to zero. The derivative is then infinitely large.
    # For now I correct this by setting the derivative value to it's max (1e308 it seems)
    # and of course print a message to warn the user
    
    # Modif    
#    if s.real ==0:
#        print('0 encountered in sqrt, set derivative to 1e308')
#        deriv = 1e308
#    else:
#        deriv = 0.5*v.df/s 
#    return Fdf(s,deriv)
Fdf.sqrt = sqrt


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
Fdf.sqrt_maystre = sqrt_maystre
    
#class Sqrt_near:
#    def __init__(self, value):
#        self.near = sqrts.Nearest(value)
#    def sqrt(self,_fdf):
#        s = self.near.sqrt(_fdf.f)
#            # Modif    
#        if s ==0:
#            print('0 encountered in sqrt, set derivative to 1e308')
#            deriv = 1e308
#        else:
#            deriv = 0.5*_fdf.df/s 
#        return Fdf(s,deriv)
#        #ORIG
#        #return Fdf(s,0.5*_fdf.df/s)

def square(v):
    return Fdf(np.square(v.f),2*v.df*v.f)
Fdf.square = square

# Mod SV
def absolute(v):
    return Fdf(np.abs(v.f),np.real(v.df*np.conjugate(v.f)))
Fdf.absolute = absolute
    
def absolute_square(v):
    return Fdf(np.square(np.real(v.f))+np.square(np.imag(v.f)),
               2*np.real(v.df*np.conjugate(v.f)))
Fdf.absolute_square = absolute_square

# I have trouble with these guys...
def sin(v):
    return Fdf(np.sin(v.f),v.df*np.cos(v.f))
Fdf.sin = sin

def cos(v):
    return Fdf(np.cos(v.f),-v.df*np.sin(v.f))
Fdf.cos = cos

def sincos(v):
    s = np.sin(v.f)
    c = np.cos(v.f)
    return Fdf(s,v.df*c),Fdf(c,-v.df*s)
Fdf.sincos = sincos
#
##Mod SV
def tan(v):    
    s = sin(v)/cos(v) #here, v is a FdF instance. 
    return s #I can directly return the fdf instance.
Fdf.tan = tan    


def exp(v):
    e = np.exp(v.f)
    return Fdf(e,v.df*e)
Fdf.exp = exp

def expi(v):
    e = np.exp(1j*v.f)
    return Fdf(e,1j*v.df*e)
Fdf.expi = expi

#modified version with exp formulas do not work better. the functions in both np.sin or np.exp overflow for large numbers...
#def sin(v):
#    r = (np.exp(1j*v.f)-np.exp(-1j*v.f))/2j
#    d = v.df*(np.exp(1j*v.f)+np.exp(-1j*v.f))/2
#    return Fdf(r,d)
#Fdf.sin = sin


def log(v):
    return Fdf(np.log(v.f), v.df/v.f)
Fdf.log = log

def newton_step(fun_fdf, r):
    fdf = fun_fdf(r)
    return (r - fdf.f/fdf.df)

def newton_ten_steps(fun_fdf, r):
    for i in range(10):
        r = newton_step(fun_fdf, r)
    rn = newton_step(fun_fdf, r)
    # Adding convergence criteria
    ii_c = 0
    while np.abs(rn - r)>2:
        ii_c = ii_c+1
        for i in range(10):
            r = newton_step(fun_fdf, r)
        rn = newton_step(fun_fdf, r)
        if ii_c>5:
            #print ('cannnot get small step. min step is %s'%np.abs(rn - r))
            break
    # Adding convergence criteria
    test = np.abs(fun_fdf(r).f)
    ii = 0
    while test>5e-5:
        for i in range(10):
            r = newton_step(fun_fdf, r)
        rn = newton_step(fun_fdf, r)
        test = np.abs(fun_fdf(r).f)
        ii = ii+1
        if ii>5:
            #print(test)
            #print ('test val fun sup to 5e-5, badly converged even after 100 steps')
            break
    # end addition 2
    return (rn, np.abs(rn - r))


