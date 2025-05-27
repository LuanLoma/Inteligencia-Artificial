import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Reproducibilidad
np.random.seed(42)
n_estudiantes = 500

# Variables predictoras sintéticas
horas_estudio = np.random.uniform(0, 8, n_estudiantes)           # horas diarias de estudio
horas_sueno = np.random.uniform(4, 10, n_estudiantes)            # horas de sueño
asistencia = np.random.uniform(60, 100, n_estudiantes)           # % de asistencia
participacion = np.random.choice(["baja", "alta"], n_estudiantes)  # participación en clase

# Variable objetivo: promedio final del estudiante (0 a 100)
promedio = (
    5 * horas_estudio
    + 1.5 * horas_sueno
    + 0.4 * asistencia
    + 5 * (participacion == "alta")
    + np.random.normal(0, 5, n_estudiantes)  # ruido
)

# Crear el DataFrame
df = pd.DataFrame({
    "Horas_Estudio": horas_estudio,
    "Horas_Sueno": horas_sueno,
    "Asistencia": asistencia,
    "Participacion": participacion,
    "Promedio_Final": promedio
})

# Codificación de la variable categórica
df["Participacion"] = df["Participacion"].map({"baja": 0, "alta": 1})

# Separar variables y objetivo
X = df.drop("Promedio_Final", axis=1)
y = df["Promedio_Final"]

# División de los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenamiento del modelo de regresión
modelo = LinearRegression()
modelo.fit(X_train_scaled, y_train)

# Predicción y evaluación
y_pred = modelo.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Evaluación del Modelo de Rendimiento Académico:")
print(f"- Error Absoluto Medio (MAE): {mae:.2f}")
print(f"- Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"- Coeficiente de Determinación (R²): {r2:.2f}")

# Visualización
plt.figure(figsize=(12, 4))

# Gráfico 1: Valores reales vs predichos
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Promedio Real")
plt.ylabel("Promedio Predicho")
plt.title("Promedio: Real vs Predicción")

# Gráfico 2: Distribución de errores
plt.subplot(1, 2, 2)
residuos = y_test - y_pred
sns.histplot(residuos, kde=True)
plt.xlabel("Error de Predicción")
plt.title("Distribución de Errores")

plt.tight_layout()
plt.show()

# Mostrar los coeficientes del modelo
print("\nImportancia de las Características:")
for nombre, coef in zip(X.columns, modelo.coef_):
    print(f"- {nombre}: {coef:.2f}")
print(f"- Intercepto: {modelo.intercept_:.2f}")

# Simulación de un nuevo estudiante
nuevo_estudiante = pd.DataFrame({
    "Horas_Estudio": [4],
    "Horas_Sueno": [7],
    "Asistencia": [90],
    "Participacion": ["alta"]
})

# Preprocesamiento y predicción
nuevo_estudiante["Participacion"] = nuevo_estudiante["Participacion"].map({"baja": 0, "alta": 1})
nuevo_estudiante_scaled = scaler.transform(nuevo_estudiante)

prediccion = modelo.predict(nuevo_estudiante_scaled)
print(f"\nPredicción de promedio para nuevo estudiante: {prediccion[0]:.2f}")
