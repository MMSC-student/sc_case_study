import numpy as np
from scipy.optimize import root

# TODO: CHECK THIS
# def newton_raphson(f, f_prime, u, tol = 1e-12):
#     # Iterating until our tolerance criteria is satified:
#     while(abs(f(u))>tol):
#         # Updating the value for the root:
#         u = u - f(u) / f_prime(u)
#     return u

def theta_scheme(f, u_n, t_n, t_n_plus_one, theta, *args):
    dt       = t_n_plus_one - t_n
    residual = lambda u_n_plus_one: u_n_plus_one - u_n - \
                                    dt * (  (1 - theta) * f(u_n, t_n, *args) 
                                          + theta * f(u_n_plus_one, t_n_plus_one, *args)
                                         )
    unew     = root(residual, u_n, method = 'krylov', tol = 1e-14).x
    return(unew)

def RK2(f, u_n, t_n, t_n_plus_one, *args):
    """
    Integrates u from u_n(t = t_n) to u_n(t = t_n_plus_one) by taking 
    slope from f, and integrates it using the RK2 method. This method 
    is second order accurate.
    
    Parameters
    ----------
    
    f: function
       Returns the slope du_dt which is 
       used to evolve u

    u_n: array
         The value of u at the beginning of the
         timestep.

    t_n: double
         Starting time

    t_n_plus_one: double
                  Ending time
    """
    dt = t_n_plus_one - t_n
    u  = u_n + f(u_n, t_n, *args) * (dt / 2)
    u  = u_n + f(u, t_n + 0.5 * dt, *args) * dt

    return(u)

def RK4(f, u_n, t_n, t_n_plus_one, *args):
    """
    Integrates u from u_n(t = t_n) to u_n(t = t_n_plus_one) by taking 
    slope from f, and integrates it using the RK4 method. This method 
    is fourth order accurate.
    
    Parameters
    ----------
    
    f: function
       Returns the slope du_dt which is 
       used to evolve u

    u_n: array
         The value of u at the beginning of the
         timestep.

    t_n: double
         Starting time

    t_n_plus_one: double
                  Ending time
    """
    dt = t_n_plus_one - t_n
    k1 = f(u_n, t_n, *args)
    u  = u_n + 0.5 * k1 * dt
    k2 = f(u, t_n + 0.5 * dt, *args)
    u  = u_n + 0.5 * k2 * dt
    k3 = f(u, t_n + 0.5 * dt, *args)
    u  = u_n + k3 * dt
    k4 = f(u, t_n + dt, *args)
    u  = u_n + ((k1 + 2 * k2 + 2 * k3 + k4) / 6) * dt

    return(u)
