import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Función
f = lambda x: np.abs(x)

def legendre_poly(n, x):
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        Pn_1 = x
        Pn_2 = np.ones_like(x)
        for i in range(2, n + 1):
            Pn = ((2 * i - 1) * x * Pn_1 - (i - 1) * Pn_2) / i
            Pn_2, Pn_1 = Pn_1, Pn
        return Pn

x = np.linspace(-1, 1, 500)
y = f(x)

# Coeficientes de la expansión de Legendre
for n in range(7):
    # Calculamos los coeficientes de la expansión usando cuadratura numérica
    coeffs = []
    for i in range(n + 1):
        Pi = lambda x: legendre_poly(i, x)
        integrand = lambda x: f(x) * Pi(x)
        ci, _ = quad(integrand, -1, 1)
        ci *= (2 * i + 1) / 2
        coeffs.append(ci)
    
    # Mostrar los coeficientes de la expansión
    print(f'Coeficientes de la expansión para n = {n}: {coeffs}')
    
    # Evaluamos
    y_fit = np.zeros_like(x)
    for i, c in enumerate(coeffs):
        y_fit += c * legendre_poly(i, x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='$|x|$', color='blue')
    plt.plot(x, y_fit, label=f'Ajuste con P_{n}(x)', linestyle='--', color='red')
    plt.title(f'Ajuste con polinomio de Legendre de grado {n}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()