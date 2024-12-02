import numpy as np
from scipy import integrate
import time
import matplotlib.pyplot as plt

serieT = 10000

def integrando(x):
    return np.log(x) / np.sqrt(np.cos(x)**2 + np.tanh(x))

# Cuadratura adaptativa 
def adaptive(f, a, b, tol, N=100000):
    class MaxIterations(Exception):
        pass

    approx = 0
    i = 0
    toli = [10 * tol]
    ai = [a]
    hi = [(b - a) / 2]
    fai = [f(a)]
    fbi = [f(b)]
    fci = [f(a + hi[i])]
    S0i = [hi[i] * (fai[i] + 4 * fci[i] + fbi[i]) / 3]
    Li = [1]

    while i >= 0:
        fd = f(ai[i] + hi[i] / 2)
        fe = f(ai[i] + 3 * hi[i] / 2)
        S1 = hi[i] * (fai[i] + 4 * fd + fci[i]) / 6
        S2 = hi[i] * (fci[i] + 4 * fe + fbi[i]) / 6
        ai_prec = ai[i]
        hi_prec = hi[i]
        fai_prec = fai[i]
        fbi_prec = fbi[i]
        fci_prec = fci[i]
        toli_prec = toli[i]
        S0i_prec = S0i[i]
        Li_prec = Li[i]

        i -= 1
        if abs(S1 + S2 - S0i_prec) < toli_prec:
            approx += S1 + S2
        else:
            if Li_prec >= N:
                raise MaxIterations("Máximo número de iteraciones alcanzado.")

            i += 1
            if i >= len(ai):
                ai.append(ai_prec + hi_prec)
                fai.append(fci_prec)
                fci.append(fe)
                fbi.append(fbi_prec)
                hi.append(hi_prec / 2)
                toli.append(toli_prec / 2)
                S0i.append(S2)
                Li.append(Li_prec + 1)
            else:
                ai[i] = ai_prec + hi_prec
                fai[i] = fci_prec
                fci[i] = fe
                fbi[i] = fbi_prec
                hi[i] = hi_prec / 2
                toli[i] = toli_prec / 2
                S0i[i] = S2
                Li[i] = Li_prec + 1

            i += 1
            if i >= len(ai):
                ai.append(ai_prec)
                fai.append(fai_prec)
                fci.append(fd)
                fbi.append(fci_prec)
                hi.append(hi[i - 1])
                toli.append(toli[i - 1])
                S0i.append(S1)
                Li.append(Li[i - 1])
            else:
                ai[i] = ai_prec
                fai[i] = fai_prec
                fci[i] = fd
                fbi[i] = fci_prec
                hi[i] = hi[i - 1]
                toli[i] = toli[i - 1]
                S0i[i] = S1
                Li[i] = Li[i - 1]

    return approx

# Precisiones
precisiones = np.array([1e-3, 1e-5, 1e-7, 1e-9])
tiempos_adaptativa = np.zeros(4)
tiempos_scipy = np.zeros(4)




# Intervalo
a, b = 0.1, 1.0

# Cálculo para cada precisión
resultados_adaptativa = np.zeros(4)
resultados_scipy = np.zeros(4)

for tol in range(4):
    # Cuadratura adaptativa
    print(tol)

    resultados_adaptativa_s = np.zeros(serieT)
    resultados_scipy_s = np.zeros(serieT)

    start = time.time()
    for i in range(serieT):
        
        resultado_adaptativa = adaptive(integrando, a, b, precisiones[tol])
        resultados_adaptativa_s[i] = resultado_adaptativa


    tiempos_adaptativa[tol] = time.time() - start

    start = time.time()
    for i in range(serieT):

        # Integrador de SciPy
        
        resultado_scipy, _ = integrate.quad(integrando, a, b, epsabs=tol, epsrel=precisiones[tol])
        resultados_scipy_s[i] = resultado_scipy
    
    tiempos_scipy[tol] = time.time() - start



    resultados_adaptativa[tol] = sum(resultados_adaptativa_s)
    resultados_scipy[tol] = sum(resultados_scipy_s)    

# Graficar resultados
plt.figure(figsize=(8, 6))
plt.loglog(precisiones, tiempos_adaptativa, label="Cuadratura Adaptativa", marker="o")
plt.loglog(precisiones, tiempos_scipy, label="Scipy Integrador", marker="s")
plt.xlabel("Precisión requerida (tol)")
plt.ylabel(f"Tiempo de cálculo (s) para {serieT} calculos")
plt.title("Comparación de tiempos de cálculo")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

tiempos_adaptativa

print (resultados_adaptativa, resultados_scipy)

# [-6170.25772224 -6169.31979931 -6169.30202529 -6169.30177108] [-6169.30176881 -6169.30176881 -6169.30176881 -6169.30176881]
#La cuadratura adaptativa ofrece buenos resultados, pero su desempeño es más lento y demanda mayor carga computacional en comparación
# con el integrador de SciPy. Por esta razón, SciPy es una opción más eficiente. 
# 
# Al inicio presente problemas con el codigo debido a que la velocidad de un calculo es extramadamente rapida independiente del metodo
# a utilizar por lo que las diferencia en tiempos depende exclusivamente de cosas de gestion de tareas del pc, lo que nos da un mal analisis
# Para solucionar esto lo que hice fue hacer que el calculo se realice una cantidad alta de veces, controlado por la variable serieT
# Por lo que el proceso se calcula un serieT veces para cada presicion, guardandose el tiempo que le toma en calcular esa cantidad de veces
# en tiempos_adaptativa y tiempos_scipy, a su vez se guarda el promedio de todas las iteraciones del valor de la integral resultados_adaptativa
# y resultados_scipy

# NOTA: 70