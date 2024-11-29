
import numpy as np
import matplotlib.pyplot as plt

# Definimos la función f(t, y) del problema
def f(t, y):
    return t * np.exp(3 * t) - 2 * y

# Solución exacta
def exact_solution(t):
    return (1/5) * t * np.exp(3 * t) - (1/25) * np.exp(3 * t) + (1/25) * np.exp(-2 * t)

# Método RK4
def rk4(f, t0, y0, t_end, h):
    t_values = [t0]
    y_values = [y0]

    t = t0
    y = y0

    while t < t_end:
        h = min(h, t_end - t)  # Ajustar el último paso si es necesario
        k1 = h * f(t, y)
        k2 = h * f(t + h / 2, y + k1 / 2)
        k3 = h * f(t + h / 2, y + k2 / 2)
        k4 = h * f(t + h, y + k3)
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += h

        t_values.append(t)
        y_values.append(y)

    return np.array(t_values), np.array(y_values)

# Método RKF (Runge-Kutta-Fehlberg)
def rkf(f, t0, y0, t_end, tol):
    t_values = [t0]
    y_values = [y0]

    # Coeficientes RKF 
    c2, c3, c4, c5, c6 = 1/4, 3/8, 12/13, 1, 1/2
    a21 = 1/4
    a31, a32 = 3/32, 9/32
    a41, a42, a43 = 1932/2197, -7200/2197, 7296/2197
    a51, a52, a53, a54 = 439/216, -8, 3680/513, -845/4104
    a61, a62, a63, a64, a65 = -8/27, 2, -3544/2565, 1859/4104, -11/40
    b1, b2, b3, b4, b5, b6 = 16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55
    b1_hat, b2_hat, b3_hat, b4_hat, b5_hat, b6_hat = 25/216, 0, 1408/2565, 2197/4104, -1/5, 0

    t = t0
    y = y0
    h = (t_end - t0) / 10  # Paso inicial

    while t < t_end:
        h = min(h, t_end - t)  

        # Calcular los k's
        k1 = h * f(t, y)
        k2 = h * f(t + c2 * h, y + a21 * k1)
        k3 = h * f(t + c3 * h, y + a31 * k1 + a32 * k2)
        k4 = h * f(t + c4 * h, y + a41 * k1 + a42 * k2 + a43 * k3)
        k5 = h * f(t + c5 * h, y + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
        k6 = h * f(t + c6 * h, y + a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)

        # Calcular las soluciones de orden 5 y 4
        y5 = y + b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6
        y4 = y + b1_hat * k1 + b2_hat * k2 + b3_hat * k3 + b4_hat * k4 + b5_hat * k5 + b6_hat * k6

        # Error estimado
        error = np.abs(y5 - y4)

        # Verificamos si el error es aceptable
        if error <= tol:
            t += h
            y = y5
            t_values.append(t)
            y_values.append(y)

        # Ajuste del paso
        h = h * min(max(0.84 * (tol / error) ** 0.25, 0.1), 4.0)

    return np.array(t_values), np.array(y_values)

# Resolvemos el problema de valor inicial con ambos métodos
t0, y0, t_end = 0, 0, 1
tol = 1e-6
h = 0.1

# Resolver con RK4
t_rk4, y_rk4 = rk4(f, t0, y0, t_end, h)

# Resolver con RKF
t_rkf, y_rkf = rkf(f, t0, y0, t_end, tol)

# Solución exacta para comparar
t_exact = np.linspace(t0, t_end, 1000)
y_exact = exact_solution(t_exact)

plt.figure(figsize=(10, 6))
plt.plot(t_exact, y_exact, label='Solución Exacta', color='black', linewidth=2)
plt.plot(t_rk4, y_rk4, 'o-', label='RK4', markersize=4)
plt.plot(t_rkf, y_rkf, 's-', label='RKF', markersize=4)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Comparación de métodos RK4 y RKF')
plt.legend()
plt.grid(True)
plt.show()