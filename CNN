CNN con Fashion MNIST – Clasificador de Prendas
Autor: Lopez Marquez Luis Angel

# Librerías necesarias
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print(f"TensorFlow v{tf.__version__}")

# Cargar datos de Fashion MNIST
dataset = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = dataset.load_data()

# Etiquetas de clases
labels = ['Camiseta', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
          'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Bota']

# Preprocesamiento
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

# Crear modelo CNN
cnn_model = keras.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation='softmax')
])

# Compilar modelo
cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

cnn_model.summary()

# Entrenamiento
history = cnn_model.fit(x_train, y_train, epochs=10, validation_split=0.15)

# Evaluación
loss, acc = cnn_model.evaluate(x_test, y_test, verbose=2)
print(f"\nExactitud en datos de prueba: {acc:.4f}")

# Gráficas de desempeño
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión Validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida Validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Predicciones
predictions = cnn_model.predict(x_test)

def show_image(idx, predictions_array, true_labels, images):
    true_lbl = true_labels[idx]
    img = images[idx].reshape(28, 28)

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_lbl = np.argmax(predictions_array)
    color = 'blue' if predicted_lbl == true_lbl else 'red'
    plt.xlabel(f"{labels[predicted_lbl]} ({100*np.max(predictions_array):.0f}%)\nReal: {labels[true_lbl]}", color=color)

def plot_probs(idx, predictions_array, true_labels):
    true_lbl = true_labels[idx]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    bars = plt.bar(range(10), predictions_array, color="#888")
    plt.ylim([0, 1])
    bars[np.argmax(predictions_array)].set_color('red')
    bars[true_lbl].set_color('blue')

# Mostrar algunas predicciones
rows, cols = 5, 3
plt.figure(figsize=(2 * 2 * cols, 2 * rows))
for i in range(rows * cols):
    plt.subplot(rows, 2 * cols, 2 * i + 1)
    show_image(i, predictions[i], y_test, x_test)
    plt.subplot(rows, 2 * cols, 2 * i + 2)
    plot_probs(i, predictions[i], y_test)
plt.tight_layout()
plt.show()
