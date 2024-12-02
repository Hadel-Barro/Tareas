import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Definir la función original
def f(x):
    return 4*x**3 - 3*x**2 + x - 5

# Puntos de interpolación iniciales
x_points = np.array([-1, 1])
y_points = f(x_points)

# Creamos una spline cúbica con los dos puntos iniciales
spline_two_points = CubicSpline(x_points, y_points)

# Generarmos más puntos para visualizar la función y la spline
x_dense = np.linspace(-1.5, 1.5, 200)
y_original = f(x_dense)
y_spline_two_points = spline_two_points(x_dense)

# Comparamos con interpolación usando 4 puntos equidistantes en el intervalo [-1, 1]
x_points_4 = np.linspace(-1, 1, 4)
y_points_4 = f(x_points_4)
spline_four_points = CubicSpline(x_points_4, y_points_4)
y_spline_four_points = spline_four_points(x_dense)

plt.figure(figsize=(10, 6))
plt.plot(x_dense, y_original, label="Función original $f(x)$", color="green")
plt.plot(x_dense, y_spline_two_points, label="Spline (2 puntos)", linestyle="--", color="orange")
plt.plot(x_dense, y_spline_four_points, label="Spline (4 puntos)", linestyle=":", color="blue")
plt.scatter(x_points, y_points, color="orange", label="Puntos usados (2 puntos)",marker="s")
plt.scatter(x_points_4, y_points_4, color="blue", label="Puntos usados (4 puntos)", marker="x")
plt.axhline(0, color="black", linewidth=0.8, linestyle=":")
plt.title("Interpolación con spline cúbica")
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.legend()
plt.grid()
plt.show()

# La diferencia es que una spline cubica si o si va a necesitar sobre 3 puntos  para construir una funcion cubica 
# que pase por todos ellos. El usar 2 puntos nos limita para reflejar la función, tener 4 puntos nos permite visualizar 
# mucho mejor como se veria la función original.

# Nota: 70