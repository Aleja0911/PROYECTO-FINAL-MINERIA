# Evidencia de consumo con datos nuevos

**Fecha de prueba:** 2026-11-16  
**Solicitud:** HTTP `POST http://127.0.0.1:8000/predict`  
**Resultado HTTP:** `200 OK`

## Procedencia y validez de la entrada

El registro es un perfil sintético preparado para esta prueba; no se extrajo del archivo Online Retail II ni pertenece a las cohortes de entrenamiento o prueba temporal. No contiene un identificador de cliente real.

Los valores representan una historia de compras hipotética: gasto histórico de 1.460 en seis pedidos, promedio de 243,3333333333 por pedido, 53 unidades, 21 productos y actividad en ocho días. Se declara una línea de devolución y ninguna cancelación o ajuste. `New Zealand` es un país real, pero no estaba entre las categorías vistas al entrenar el codificador; permite comprobar que el pipeline acepta categorías nuevas mediante `handle_unknown="ignore"`. Todos los campos y valores cumplen el esquema de entrada de la API.

## JSON enviado

```json
{
  "historical_spend": 1460.0,
  "order_count": 6,
  "units_purchased": 53,
  "distinct_products": 21,
  "active_days": 8,
  "days_since_last_purchase": 3,
  "mean_order_spend": 243.3333333333,
  "historical_return_lines": 1,
  "historical_cancellation_lines": 0,
  "historical_adjustment_lines": 0,
  "country": "New Zealand"
}
```

## Respuesta recibida

```json
{
  "prediction": 2202.7791226811255,
  "target": "future_spend",
  "model": "Ridge"
}
```

## Interpretación y verificación del artefacto

El pipeline estima un gasto futuro elegible de aproximadamente **2.202,78 unidades monetarias** para el horizonte definido por la preparación del modelo. Es una predicción, no un gasto observado ni una garantía del resultado futuro.

La API reportó `Ridge`. Además, se envió el mismo perfil directamente al pipeline recuperado de `05_MACHINE_LEARNING/modelo.pkl`: produjo `2202.7791226811255`, coincidente con la respuesta HTTP dentro de una tolerancia de `1e-12`. Esto comprueba que la respuesta del endpoint corresponde al artefacto serializado y a su preprocesamiento integrado.
