import numpy as np
import matplotlib.pyplot as plt

# Datos iniciales
mu_X = np.array([1, 2])  # Media de X
Sigma_X = np.array([[2, 1], [1, 2]])  # Matriz de covarianza de X

# Transformación
A = np.array([[1, 2], [3, 4]])  # Matriz de transformación
b = np.array([1, 1])  # Vector de traslación

# Cálculo de la media de Y
mu_Y = A @ mu_X + b

# Cálculo de la covarianza de Y
Sigma_Y = A @ Sigma_X @ A.T

mu_Y, Sigma_Y

def plot_ellipse(mean, cov, color='blue', label=None):
    # Autovalores y autovectores
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Ordenar autovalores de mayor a menor
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    
    # Ángulo de rotación de la elipse
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    angle = np.degrees(angle)
    
    # Tamaño de los ejes
    width, height = 2 * np.sqrt(eigenvalues)  
    
    # Graficamos la elipse
    ellipse = plt.matplotlib.patches.Ellipse(mean, width, height, angle=angle, 
                                             edgecolor=color, fill=False, label=label)
    plt.gca().add_patch(ellipse)


plt.figure(figsize=(8, 8))

# Contornos de X
plot_ellipse(mu_X, Sigma_X, color='orange', label='Original (X)')

# Contornos de Y
plot_ellipse(mu_Y, Sigma_Y, color='cyan', label='Transformación (Y)')

plt.scatter(*mu_X, color='orange', label='Media X')
plt.scatter(*mu_Y, color='cyan', label='Media Y')
plt.axhline(0, color='black', linewidth=0.5, linestyle='-')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()
plt.grid(alpha=0.3)
plt.title("Contornos de Equiprobabilidad de X e Y")
plt.xlabel("Eje 1")
plt.ylabel("Eje 2")
plt.axis('equal')
plt.show()

#La matriz A va a afectar la orientación de los ejes principales de la 
#elipse, el tamaño y forma del contorno
#El vector b no cambiara la geometria, solo desplazara la elipse.

#NOTA: 70
