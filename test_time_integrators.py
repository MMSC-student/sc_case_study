import numpy as np
from time_integrators import theta_scheme, RK2, RK4
import pylab as pl
pl.style.use('latexplot')

# The function is of the form du / dt = f(u, t)
def f(u, t):
    return u

def test_theta():
    N_x = 100
    N_t = 2**np.arange(5, 11)
    u0  = np.random.rand(N_x)
    err = np.zeros(N_t.size)
    for i in range(N_t.size):
        t  = np.linspace(0, 1, N_t[i] + 1)
        u  = u0
        for time_index in range(N_t[i]):
            u = theta_scheme(f, u, t[time_index], t[time_index + 1], 0.5)
        err[i] = np.linalg.norm(u - u0 * np.exp(1))

    pl.loglog(N_t, err, '-o', label = 'Numerical')
    pl.loglog(N_t, err[0] * (N_t[0]/N_t)**2, 'k--', label = r'$\mathcal{O}((\Delta t)^2)$')
    pl.ylabel('Error')
    pl.xlabel(r'$N_t$')
    pl.legend()
    pl.grid()
    pl.savefig('theta.png', bbox_inches = 'tight')
    pl.clf()

    assert(abs(np.polyfit(np.log(N_t), np.log(err), 1)[0] + 2) < 0.1)

def test_RK2():
    N_x = 100
    N_t = 2**np.arange(5, 11)
    u0  = np.random.rand(N_x)
    err = np.zeros(N_t.size)
    for i in range(N_t.size):
        t  = np.linspace(0, 1, N_t[i] + 1)
        u  = u0
        for time_index in range(N_t[i]):
            u = RK2(f, u, t[time_index], t[time_index + 1])
        err[i] = np.linalg.norm(u - u0 * np.exp(1))

    assert(abs(np.polyfit(np.log(N_t), np.log(err), 1)[0] + 2) < 0.1)

def test_RK4():
    N_x = 100
    N_t = 2**np.arange(5, 11)
    u0  = np.random.rand(N_x)
    err = np.zeros(N_t.size)
    for i in range(N_t.size):
        t  = np.linspace(0, 1, N_t[i] + 1)
        u  = u0
        for time_index in range(N_t[i]):
            u = RK4(f, u, t[time_index], t[time_index + 1])
        err[i] = np.linalg.norm(u - u0 * np.exp(1))

    pl.loglog(N_t, err, '-o', label = 'Numerical')
    pl.loglog(N_t, err[0] * (N_t[0]/N_t)**4, 'k--', label = r'$\mathcal{O}((\Delta t)^4)$')
    pl.ylabel('Error')
    pl.xlabel(r'$N_t$')
    pl.legend()
    pl.grid()
    pl.savefig('rk4.png', bbox_inches = 'tight')

    assert(abs(np.polyfit(np.log(N_t), np.log(err), 1)[0] + 4) < 0.1)
