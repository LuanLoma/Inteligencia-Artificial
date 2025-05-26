import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# AUTOR: Lopez Marquez Luis Angel

# VARIABLES DE ENTRADA
resp_0a100 = ctrl.Antecedent(np.arange(1.4, 4.7, 0.1), 'resp_0a100')
resp_0a100['lenta'] = fuzz.trapmf(resp_0a100.universe, [3.6, 4.0, 4.6, 4.6])
resp_0a100['normal'] = fuzz.trimf(resp_0a100.universe, [2.6, 3.2, 3.8])
resp_0a100['muy_rapida'] = fuzz.trapmf(resp_0a100.universe, [1.4, 1.4, 2.2, 2.8])

vuelta_pista = ctrl.Antecedent(np.arange(6.4, 8.2, 0.1), 'vuelta_pista')
vuelta_pista['pobre'] = fuzz.trapmf(vuelta_pista.universe, [7.6, 7.9, 8.1, 8.1])
vuelta_pista['aceptable'] = fuzz.trimf(vuelta_pista.universe, [6.9, 7.3, 7.7])
vuelta_pista['rápido'] = fuzz.trapmf(vuelta_pista.universe, [6.4, 6.4, 6.9, 7.2])

masa_total = ctrl.Antecedent(np.arange(980, 2520, 10), 'masa_total')
masa_total['liviano'] = fuzz.trapmf(masa_total.universe, [980, 980, 1180, 1350])
masa_total['moderado'] = fuzz.trimf(masa_total.universe, [1300, 1550, 1800])
masa_total['pesado'] = fuzz.trapmf(masa_total.universe, [1700, 1900, 2520, 2520])

# VARIABLE DE SALIDA
rendimiento = ctrl.Consequent(np.arange(0, 101, 1), 'rendimiento')
rendimiento['bajo'] = fuzz.trimf(rendimiento.universe, [0, 25, 45])
rendimiento['medio'] = fuzz.trimf(rendimiento.universe, [35, 55, 75])
rendimiento['alto'] = fuzz.trimf(rendimiento.universe, [60, 78, 92])
rendimiento['premium'] = fuzz.trimf(rendimiento.universe, [80, 92, 100])

# REGLAS DIFUSAS
reglas = [
    ctrl.Rule(resp_0a100['muy_rapida'] & vuelta_pista['rápido'], rendimiento['premium']),
    ctrl.Rule(masa_total['liviano'] & resp_0a100['normal'], rendimiento['alto']),
    ctrl.Rule(vuelta_pista['pobre'] | masa_total['pesado'], rendimiento['bajo']),
    ctrl.Rule(resp_0a100['normal'] & vuelta_pista['aceptable'], rendimiento['medio']),
    ctrl.Rule(resp_0a100['lenta'], rendimiento['bajo']),
    ctrl.Rule(vuelta_pista['rápido'] & masa_total['moderado'], rendimiento['alto']),
    ctrl.Rule(masa_total['moderado'] & resp_0a100['normal'], rendimiento['medio']),
]

# PUNTUACIÓN NUMÉRICA
def puntuacion_extra(tipo_motor, caja, traccion, consumo, unidad):
    score = 0
    if tipo_motor == "Eléctrico":
        score += 10
    elif tipo_motor == "Híbrido":
        score += 15
    else:
        score += 5

    score += 10 if caja == "Manual" else 5
    score += 5 if traccion == "AWD" else 2

    if unidad == "kWh/100km":
        score += 10 if consumo < 20 else 5 if consumo < 30 else 0
    else:
        score += 10 if consumo < 12 else 5 if consumo < 20 else 0

    return score

# FUNCIÓN PRINCIPAL DE EVALUACIÓN
def evaluar_vehiculo(*params):
    _, acc, vuelta, peso, mot, caja, trac, cons, und = params

    acc = np.clip(acc, 1.4, 4.6)
    vuelta = np.clip(vuelta, 6.4, 8.1)
    peso = np.clip(peso, 980, 2520)

    sistema = ctrl.ControlSystem(reglas)
    simulacion = ctrl.ControlSystemSimulation(sistema)

    try:
        simulacion.input['resp_0a100'] = acc
        simulacion.input['vuelta_pista'] = vuelta
        simulacion.input['masa_total'] = peso
        simulacion.compute()
        valor_difuso = simulacion.output['rendimiento']
    except:
        valor_difuso = 0

    valor_numerico = puntuacion_extra(mot, caja, trac, cons, und)
    return min(100, max(0, (valor_difuso * 0.7) + (valor_numerico * 0.75)))

# DATOS DE VEHÍCULOS
vehiculos = {
    "Velon X1": (350, 2.3, 6.7, 1490, "Híbrido", "Automático", "AWD", 13, "L/100km"),
    "Electra R": (390, 1.9, 7.0, 2200, "Eléctrico", "Automático", "AWD", 28, "kWh/100km"),
    "Stratus V12": (310, 3.1, 6.85, 1420, "Combustión", "Manual", "RWD", 17, "L/100km")
}

resultados = {nombre: evaluar_vehiculo(*datos) for nombre, datos in vehiculos.items()}

# GRAFICAR RESULTADOS
plt.figure(figsize=(12, 6))
for label in rendimiento.terms:
    plt.plot(rendimiento.universe,
             fuzz.interp_membership(rendimiento.universe, rendimiento[label].mf, rendimiento.universe),
             linewidth=2,
             label=label.capitalize())

colores = ['#D62728', '#2CA02C', '#1F77B4']
for (nombre, valor), color in zip(resultados.items(), colores):
    plt.axvline(valor, color=color, linestyle='--', linewidth=3,
                label=f'{nombre} ({valor:.1f})')

plt.title('Comparativa de Rendimiento Vehicular')
plt.xlabel('Puntuación Total (0-100)')
plt.ylabel('Grado de Pertenencia')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.xlim(0, 100)
plt.ylim(0, 1)
plt.tight_layout()

# Imprimir resultados
print("\n--- PUNTUACIÓN FINAL ---")
for modelo, valor in resultados.items():
    print(f"{modelo:<15}: {valor:.1f}/100")
print()

plt.show()
