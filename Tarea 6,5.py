import numpy as np
import pandas as pd

# Datos proporcionados
sigma = np.array([0.75, 0.77, 0.79, 0.81, 0.83])
n_values = np.array([1.1324e-3, 1.1376e-3, 1.1386e-3, 1.1454e-3, 1.1474e-3])
h = sigma[1] - sigma[0]  # Paso 

# Derivada a tres puntos en σ = 0.79 (índice 2)
f_prime = (n_values[3] - n_values[1]) / (2 * h)  # Fórmula de tres puntos

# Derivada tercera a cinco puntos en σ = 0.79
f_third = (-n_values[4] + 2 * n_values[3] - 2 * n_values[1] + n_values[0]) / (2 * h**3)

# Cálculo del error
error = -(h**2 / 6) * f_third

# Ruido
delta_n = 1e-6

print("Resultados:")
print(f"Paso h: {h}")
print(f"Derivada (f') en σ=0.79: {f_prime}")
print(f"Tercera derivada (f''') en σ=0.79: {f_third}")
print(f"Error estimado: {error}")
print(f"Ruido Δn: {delta_n}")

# No creo que sea necesario cambiar el paso, segun los resultados se ve bien el que estamos usando que es 0.02

# NOTA: 70