import tensorflow as tf
from jax import grad, random, jit
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt

# Cargar el dataset de Fashion MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalizarmos los datos
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = jnp.array(train_images.reshape(-1, 28 * 28))
test_images = jnp.array(test_images.reshape(-1, 28 * 28))
train_labels = jnp.array(train_labels)
test_labels = jnp.array(test_labels)

# ReLU
@jit
def relu(z):
    return jnp.maximum(0, z)

# Softmax
@jit
def softmax(z):
    exp_z = jnp.exp(z)
    return exp_z / jnp.sum(exp_z, axis=-1, keepdims=True)

# MLP
def mlp(params, x):
    w1, b1, w2, b2 = params
    z1 = jnp.dot(x, w1) + b1
    a1 = relu(z1)
    z2 = jnp.dot(a1, w2) + b2
    return softmax(z2)

# Inicialización de pesos y sesgos
def inicializar_pesos(rng, input_size, hidden_size, output_size):
    w1 = random.normal(rng, (input_size, hidden_size)) * 0.001
    b1 = jnp.zeros((hidden_size,))
    w2 = random.normal(rng, (hidden_size, output_size)) * 0.001
    b2 = jnp.zeros((output_size,))
    return w1, b1, w2, b2


input_size = 28 * 28
hidden_size = 128
output_size = 10

# Inicializamos parámetros
rng = random.PRNGKey(0)
params = inicializar_pesos(rng, input_size, hidden_size, output_size)  # Reinicializar los parámetros para cada optimizador

# Función de pérdida 
@jit
def cross_entropy_loss(params, x, y):
    preds = mlp(params, x)
    return -jnp.mean(jnp.sum(y * jnp.log(preds), axis=-1))

# Precisión
@jit
def accuracy(params, x, y):
    predictions = jnp.argmax(mlp(params, x), axis=1)
    return jnp.mean(predictions == y)

# Función para entrenar la red
def train_step(params, x, y, opt_state, optimizer):
    grads = grad(cross_entropy_loss)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# Configurar los algoritmos de optimización
optimizers = {
    'Momentum': optax.sgd(learning_rate=0.0001, momentum=0.9),
    'RMSProp': optax.rmsprop(learning_rate=0.0001, eps=1e-6),
    'Adam': optax.adam(learning_rate=0.0001, eps=1e-7)
}

# Entrenamiento para cada optimizador
results = {}
for name, optimizer in optimizers.items():
    params = inicializar_pesos(rng, input_size, hidden_size, output_size)
    print(f"Entrenando con {name}...")
    opt_state = optimizer.init(params)
    accuracies = []
    epochs = 20
    batch_size = 64

    for epoch in range(epochs):
        num_batches = len(train_images) // batch_size
        for i in range(num_batches):
            x_batch = train_images[i * batch_size:(i + 1) * batch_size]
            y_batch = jnp.eye(10)[train_labels[i * batch_size:(i + 1) * batch_size]]
            params, opt_state = train_step(params, x_batch, y_batch, opt_state, optimizer)
        
        test_acc = accuracy(params, test_images, test_labels)
        accuracies.append(test_acc)
        print(f"{name} - Época {epoch + 1}, Precisión en test: {test_acc:.3f}")
    
    results[name] = accuracies

# Graficar los resultados
for name, accuracies in results.items():
    plt.plot(np.arange(len(accuracies)) + 1, accuracies, label=name)

plt.xlabel("Época")
plt.ylabel("Precisión en el conjunto")
plt.legend()
plt.show()

#Nota: 60