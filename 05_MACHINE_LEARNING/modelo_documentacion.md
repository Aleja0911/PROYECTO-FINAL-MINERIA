# Artefacto del modelo

## Archivo y reconstrucción

- Artefacto: `modelo.pkl`
- Formato: diccionario Python serializado con `joblib`; incluye el pipeline ajustado y metadatos.
- Regenerar y verificar: `python guardar_modelo.py` desde el entorno listado abajo. El script vuelve a ejecutar `preparacion_modelo.ipynb`, entrena conforme a su corte temporal, guarda el pipeline seleccionado, lo recarga y comprueba que las predicciones se mantengan.
- El estimador guardado es el pipeline completo de Ridge: imputación/escalado de variables numéricas, imputación/codificación one-hot del país y regresión. Está ajustado únicamente con la cohorte temporal de entrenamiento; no se reentrena con el conjunto de prueba.

## Entradas

Se espera un `pandas.DataFrame` con una fila por cliente/corte y estas columnas, calculadas con las mismas reglas y ventana de observación del notebook de preparación:

| Columna | Tipo esperado | Descripción |
|---|---|---|
| `historical_spend` | numérico | Suma histórica de importes elegibles. |
| `order_count` | numérico | Pedidos elegibles distintos. |
| `units_purchased` | numérico | Unidades compradas en líneas elegibles. |
| `distinct_products` | numérico | Productos distintos comprados. |
| `active_days` | numérico | Días con compras elegibles. |
| `days_since_last_purchase` | numérico | Días desde la última compra elegible hasta el corte. |
| `mean_order_spend` | numérico | Gasto elegible promedio por pedido. |
| `historical_return_lines` | numérico | Líneas históricas con cantidad negativa. |
| `historical_cancellation_lines` | numérico | Líneas históricas marcadas como cancelación. |
| `historical_adjustment_lines` | numérico | Líneas históricas marcadas como ajuste. |
| `country` | texto | País más frecuente del cliente en compras elegibles. |

`customer_id` es el identificador de la unidad, pero no es predictor y no debe incluirse en el DataFrame. El pipeline imputa valores numéricos faltantes, imputa el país faltante y tolera países no vistos durante el entrenamiento.

## Salida

Una predicción por fila: importe futuro no negativo (`future_spend`) en las unidades monetarias de la columna `Price`. Para reproducir el posprocesamiento usado en la evaluación, las predicciones negativas se limitan a cero:

```python
import joblib
import numpy as np

artifact = joblib.load("modelo.pkl")
X = ...  # DataFrame con el esquema indicado
predicted_spend = np.maximum(artifact["estimator"].predict(X), 0.0)
```

Los metadatos de versiones, ventanas temporales, criterio de selección y métricas de comparación están en `artifact["metadata"]`.

## Dependencias y versiones de generación

Dependencias declaradas en `requirements.txt`: Python, pandas, NumPy, scikit-learn, joblib, ipykernel y nbclient. El entorno usado al generar y verificar el archivo registró:

- Python 3.14.7
- pandas 3.1.1
- NumPy 2.5.5
- scikit-learn 1.9.1
- joblib 1.6.0

Las versiones también se guardan en los metadatos del artefacto. Para reproducibilidad, cargarlo con las versiones registradas o regenerarlo con `guardar_modelo.py` si cambia el entorno. Los pickles/joblib pueden ejecutar código al cargarse: usar únicamente este archivo local confiable, no artefactos de origen desconocido.

## Alcance

El objetivo es un proxy de gasto transaccional elegible, no ventas netas contables ni margen. La evaluación temporal se utilizó para comparar y seleccionar el modelo, así que las métricas registradas no son una evaluación final independiente. Validar en un periodo futuro no usado antes de tomar decisiones financieras individuales.
