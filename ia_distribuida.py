import numpy as np

# ----------------------------
# Datos de cada nodo
# ----------------------------

nodos = []

for i in range(3):
    X = np.random.rand(100, 1) * 10
    y = 3 * X + 2 + np.random.randn(100, 1)
    nodos.append((X, y))

# ----------------------------
# Modelo
# ----------------------------

def entrenar_local(X, y, epochs=100, lr=0.001):
    w = np.random.randn(1)
    b = np.random.randn(1)

    n = len(X)

    for _ in range(epochs):
        y_pred = X.flatten() * w + b

        dw = (-2/n) * np.sum(X.flatten() * (y.flatten() - y_pred))
        db = (-2/n) * np.sum(y.flatten() - y_pred)

        w -= lr * dw
        b -= lr * db

    return w, b

# ----------------------------
# Entrenamiento distribuido
# ----------------------------

pesos = []
biases = []

for i, (X, y) in enumerate(nodos):
    w, b = entrenar_local(X, y)

    print(f"Nodo {i+1}")
    print(f"Peso = {w.item():.4f}")
    print(f"Bias = {b.item():.4f}")

    pesos.append(w)
    biases.append(b)

# ----------------------------
# Agregación central
# ----------------------------

w_global = np.mean(pesos)
b_global = np.mean(biases)

print("Modelo global")
print(f"Peso promedio = {w_global:.4f}")
print(f"Bias promedio = {b_global:.4f}")

# ----------------------------
# Predicción
# ----------------------------

x_nuevo = 8

pred = w_global * x_nuevo + b_global

print(f"\nPredicción para x={x_nuevo}: {pred:.2f}")