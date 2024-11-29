def prob(evento, espacio):
    """
    Calcula la probabilidad de que ocurra un 'evento' dado un 'espacio' muestral.
    """
    return len(evento) / len(espacio)

# Creamos el espacio muestral con reemplazo
urna = ['roja'] * 10 + ['no_roja'] * (23 - 10)  # Definimos las bolas en la urna
extracciones = 4

from itertools import product

espacio_muestral = set(product(urna, repeat=extracciones))

# Definimos el evento que nos interesa, todas las bolas extraídas son rojas
todas_rojas = {e for e in espacio_muestral if all(bola == 'roja' for bola in e)}

probabilidad = prob(todas_rojas, espacio_muestral)

print(f"La probabilidad de que todas las bolas extraídas sean rojas: {probabilidad:.5f}")
