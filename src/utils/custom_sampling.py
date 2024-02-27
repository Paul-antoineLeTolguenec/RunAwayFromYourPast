import numpy as np 



def affine_sample(batch_size, a = 1.0):
    """
    Affine transformation of the uniform distribution. 
    f(x) = ax+b
    """
    # sample from the uniform distribution
    x = np.random.rand(batch_size)
    # affine transformation
    b = 1.0 - a/2.0
    y = (-b + np.sqrt(b**2 + 2.0*a*x))/a
    return y

def exp_dec(batch_size, tau = 0.05):
    """
    Exponential transformation of the uniform distribution. 
    f(x) = k*exp(x/tau)
    tau ref = 0.05
    """
    # sample from the uniform distribution
    x = np.random.rand(batch_size)
    # valid k 
    k = 1/(tau*(np.exp(1/tau)-1))
    # exponential decay
    y = np.log(x/(k*tau)+1)*tau
    return y

def tau_charac(tau,percentage=0.98):
    """
    Characteristic time of the exponential distribution. 
    """
    return -np.log(1 - percentage)*tau