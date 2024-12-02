import numpy as np
from scipy.stats import entropy

# Función para calcular la entropía marginal
def calcular_entropia_marginal(p):
    return entropy(p, base=2)

# Función para calcular la entropía conjunta
def calcular_entropia_conjunta(p_xy):
    p_flat = p_xy.flatten()  
    return entropy(p_flat, base=2)


p_x = np.array([0.5, 0.5])  # Probabilidades marginales de x
p_y = np.array([0.4, 0.6])  # Probabilidades marginales de y


p_xy = np.outer(p_x, p_y)

# Cálculos
H_x = calcular_entropia_marginal(p_x)
H_y = calcular_entropia_marginal(p_y)
H_xy = calcular_entropia_conjunta(p_xy)

# Verificaciones
resultado_1 = np.isclose(H_xy, H_x + H_y)  
resultado_2 = H_x <= H_xy  

print(H_x, H_y, H_xy, resultado_1, resultado_2)

# En el caso que no sean independientes, el primero sería False. La segunda siempre se cumple.

#Nota: 70 Se cumple en ambos casos