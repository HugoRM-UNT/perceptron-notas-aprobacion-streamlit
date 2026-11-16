# Procedimiento del proyecto: perceptrón para predecir la aprobación de una experiencia curricular

**Asignatura:** Sistemas Inteligentes y Machine Learning
**Modalidad:** Presentación individual
**Estudiante:** Hugo Ernesto Rebaza Mendoza

---

## 1. Objetivo

Entrenar un perceptrón simple (una neurona) que determine si un estudiante aprueba o no una experiencia curricular, según las reglas del ejercicio:

- Se consideran 3 promedios (3 unidades), cada uno entre 0 y 20.
- Se aprueba con al menos 2 unidades aprobadas.
- No se calcula un promedio final: solo se aplica esa regla.

La nota aprobatoria por unidad es **14**.

## 2. Herramientas

Python, NumPy, pandas y Streamlit, con un entorno virtual (`venv`). El perceptrón está programado directamente con NumPy, sin bibliotecas de machine learning, siguiendo el algoritmo visto en clase.

## 3. Procedimiento

### Paso 1. Formulación del problema
Es un problema de **aprendizaje supervisado de clasificación binaria**: cada patrón (3 promedios) tiene una salida deseada `yd` (1 = aprueba, 0 = desaprueba).

### Paso 2. Modelo
Neurona con 3 entradas, 3 pesos y un sesgo (4 parámetros entrenables):

```
NET = x1·w1 + x2·w2 + x3·w3 + 1·wb
y   = 1 si NET ≥ 0, y = 0 si NET < 0     (función escalón)
```

### Paso 3. Regla de aprendizaje
Regla delta, igual que en el ejemplo de clase:

```
error = yd − y
Wi(t+1) = Wi(t) + fa · error · xi
wb(t+1) = wb(t) + fa · error · 1
```

Si en una época completa no hay errores, el entrenamiento termina porque ya se encontraron los pesos.

### Paso 4. Generación de datos
No existe un conjunto de datos externo, por lo que se generaron **400 estudiantes sintéticos** con semilla 42. Cada uno tiene 3 promedios enteros entre 0 y 20 (distribución uniforme). La salida deseada se calcula con la regla del ejercicio.

Los datos se separaron al azar en:

| Conjunto | Estudiantes | Uso |
|---|---|---|
| Entrenamiento | 240 | Ajustar los pesos |
| Validación | 80 | Comparar configuraciones |
| Prueba (test) | 80 | Evaluación final, una sola vez |

El análisis exploratorio mostró que las clases están desbalanceadas: aprueba cerca del 26 % de los estudiantes. Por eso se compara siempre contra una línea base (predecir siempre "desaprueba").

### Paso 5. Preprocesamiento
Se probaron dos formas de alimentar la red:

1. **Binaria:** cada nota se convierte en 1 si es ≥ 14 (unidad aprobada) o en 0 si no lo es.
2. **Normalizada:** cada nota se divide entre 20, quedando en el rango [0, 1].

### Paso 6. Entrenamiento del modelo de referencia
Pesos iniciales de clase (w = [0.5, 0.4, 0.3], wb = 1.5) y fa = 0.5. Se registró el cálculo paso a paso de la primera época (NET, y, error, nuevos pesos), como en el ejemplo de la diapositiva.

### Paso 7. Experimentos
Se compararon representación (binaria o normalizada), factor de aprendizaje (0.1, 0.5 y 1.0) y pesos iniciales (fijos o aleatorios), con 10 semillas por configuración y un máximo de 50 épocas. La selección se hizo **solo con la exactitud de validación**; en caso de empate, se eligió la de menos épocas.

### Paso 8. Evaluación final
La configuración elegida se entrenó con el conjunto de entrenamiento y se evaluó **una sola vez** en el conjunto de prueba, con exactitud, precisión, recall, F1 y matriz de confusión.

### Paso 9. Aplicación en Streamlit
Se desarrolló una aplicación con seis pestañas: problema y teoría, datos, entrenamiento, experimentos, evaluación final con demostración, y reproducibilidad con conclusiones.

## 4. Resultados

Valores obtenidos con la configuración por defecto (semilla 42, 400 estudiantes, nota 14):

| Representación | Convergencia | Épocas | Exactitud en validación |
|---|---|---|---|
| Binaria | 100 % de las corridas | 2 | 100 % |
| Normalizada | 0 % de las corridas (tope de 50 épocas) | 50 | entre 91.8 % y 92.8 % según la configuración |

- **Línea base** (predecir siempre "desaprueba"): 75 % en validación y 70 % en prueba.
- **Configuración elegida:** representación binaria, fa = 0.1, pesos iniciales fijos.
- **Pesos finales:** w = [0.1, 0.2, 0.2], wb = −0.2, tras 2 épocas.
- **Evaluación en prueba:** 100 % de exactitud, precisión, recall y F1.

## 5. Interpretación y limitaciones

- Con entradas binarias el problema es **linealmente separable**, por lo que el perceptrón converge y reproduce exactamente la regla. Por eso la exactitud de 100 % es esperable: las etiquetas se calculan con esa misma regla.
- Con notas normalizadas la regla depende de un umbral por unidad y **no es linealmente separable**; una sola neurona solo la aproxima. Resolverla exactamente con notas completas requeriría una red multicapa.
- Los datos son sintéticos (notas uniformes), no representan la distribución real de un salón de clases.
- Con fa = 0.1 los pesos finales dejan algunos NET muy cerca de 0. Clasifica bien, pero el perceptrón no garantiza un margen entre clases.

## 6. Cómo ejecutar

**Windows**
```bash
cd perceptron_streamlit
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**Mac / Linux**
```bash
cd perceptron_streamlit
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Orden de uso: barra lateral con los valores indicados → pestaña 3 (entrenar) → pestaña 4 (experimentos) → pestaña 5 (evaluación final y demostración).

## 7. Estructura del proyecto

```
perceptron_streamlit/
├── app.py              # interfaz Streamlit
├── perceptron.py       # perceptrón, datos, métricas y experimentos
├── requirements.txt
├── README.md
└── .streamlit/config.toml
```

## 8. Referencias

- Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., … Oliphant, T. E. (2020). Array programming with NumPy. *Nature, 585*(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2
- Minsky, M., & Papert, S. (1969). *Perceptrons: An introduction to computational geometry*. MIT Press.
- Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review, 65*(6), 386–408. https://doi.org/10.1037/h0042519
