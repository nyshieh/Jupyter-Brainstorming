import matplotlib.pyplot as plt
import numpy as np
import hankel

# We need the spline interpolation to obtain a smooth function that can be transformed back to the original function
from scipy.interpolate import InterpolatedUnivariateSpline as spline

realspace = np.linspace(0, 100, 300)[1:]  #  Realspace domain
hankelspace = np.logspace(-5, 15, 300) #  Hankel space domain
ht = hankel.HankelTransform(nu=1, N=500, h=0.0005)

gupFilter = np.genfromtxt('gup1997J147pt.csv', float, delimiter=',').flatten()
print(f"n={len(gupFilter)} Filter loaded")
print(gupFilter)

def abscissaeJ1(r):
    # Returns the lambdas used in the method of Guptasarma and Singh 1997
    # function f(lambda) is evaluated at these lambdas

    # 47 point filter parameters:
    a = -3.05078187595e0
    s = 1.10599010095e-1
    # 140 point filter parameters
    # a = -7.91001919000e0
    # s = 8.79671439570e-2
    rangefunc = np.arange(0, len(gupFilter))

    eval_points = (1 / r[:,np.newaxis]) * np.power(10, a + (rangefunc * s))

    return eval_points

def filteredHankel(r, ufunc):
    # Calculates the J1 Hankel Transform with a 47 point filter
    # r: points to calculate the transform at
    # ufunc: callable function to be hankel transformed
    lambdas = abscissaeJ1(r)

    K = ufunc(lambdas)

    prod = np.multiply(gupFilter, K).sum(axis=1) / r
    return prod


# Hankel Transform Functions
tf8 = lambda r: r / np.power(1 + np.power(r, 2), 1.5)
tf9 = lambda r: r * np.exp(-(np.power(r, 2) / 4)) / 4
tf10 = lambda r: (np.sqrt(np.power(r, 2) + 1) - 1) / r / np.sqrt(np.power(r, 2) + 1)

f8 = lambda l: l * np.exp(-l)
f9 = lambda l: np.power(l,2) * np.exp(-1* np.power(l,2))
f10 = lambda l: np.exp(-l)

# Input function to test
in_func = tf10
orig_func = f10
# Forward Hankel Testing
filterHankel = filteredHankel(hankelspace, in_func)
analyticalHankel = in_func(hankelspace)
libraryHankel = ht.transform(orig_func, hankelspace, ret_err=False)

# Interpolate spline for to convert points into functions
inversespline_filter = spline(hankelspace, filterHankel)
inversespline_library = spline(hankelspace, libraryHankel)

# Carry out inverse transform
inverse_filter = filteredHankel(realspace, inversespline_filter)
inverse_library = ht.transform(inversespline_library, realspace, inverse=True, ret_err=False)

# Left side is using Guptasarma Filters
plt.figure(figsize=(8,10), dpi=200, facecolor='w', edgecolor='k')
plt.subplot(311)
plt.plot(hankelspace, filterHankel, label='Guptasarma 140 point filter')
plt.plot(hankelspace, analyticalHankel, "r--", label='Analytical Hankel Function')
plt.plot(hankelspace, libraryHankel, "g--", label='Library Transformed Hankel Function')
plt.xlim([0, 25])
plt.legend()
plt.title('Hankel Transform Comparison')

plt.subplot(312)
plt.title('Hankel Transform Error')
plt.plot(hankelspace, abs(filterHankel-analyticalHankel), "r--")
plt.plot(hankelspace, abs(libraryHankel-analyticalHankel), "g--")
plt.xlim([0, 25])

plt.subplot(313)
plt.title('Inverse Transform')
plt.plot(realspace, inverse_filter)
plt.plot(realspace, orig_func(realspace), 'r')
plt.plot(realspace, inverse_library, 'g.')
plt.xlim([0, 25])
plt.show()
