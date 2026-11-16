# API de predicción con FastAPI

API local para predecir el gasto futuro elegible de clientes mediante el pipeline guardado en `05_MACHINE_LEARNING/modelo.pkl`.

## Instalación y ejecución

Desde la raíz del proyecto, en PowerShell:

```powershell
.\.venv-1\Scripts\python.exe -m pip install -r 06_API_FASTAPI\requirements.txt
.\.venv-1\Scripts\python.exe -m uvicorn main:app --app-dir 06_API_FASTAPI --reload
```

La API busca el modelo en `05_MACHINE_LEARNING\modelo.pkl`. Para usar otra ubicación, define `MODEL_PATH` antes de iniciar Uvicorn:

```powershell
$env:MODEL_PATH = "C:\ruta\confiable\modelo.pkl"
```

Documentación interactiva: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Rutas

- `GET /`: información general y rutas principales.
- `GET /health`: devuelve `200` si el modelo cargó; devuelve `503` si el servicio está levantado pero el modelo no está disponible.
- `POST /predict`: valida las variables de entrada y devuelve el gasto futuro estimado, limitado a cero como mínimo.

## Ejemplo de predicción

En Swagger UI, abrir `POST /predict` y enviar:

```json
{
  "historical_spend": 500.0,
  "order_count": 4,
  "units_purchased": 20,
  "distinct_products": 8,
  "active_days": 3,
  "days_since_last_purchase": 2,
  "mean_order_spend": 125.0,
  "historical_return_lines": 0,
  "historical_cancellation_lines": 0,
  "historical_adjustment_lines": 0,
  "country": "United Kingdom"
}
```

Respuesta:

```json
{
  "prediction": 958.45,
  "target": "future_spend",
  "model": "Ridge"
}
```

Las diez variables numéricas y `country` deben corresponder a las definiciones y ventana histórica descritas en `05_MACHINE_LEARNING\modelo_documentacion.md`. No se envía `customer_id`: es un identificador, no una característica del modelo. Las entradas adicionales, cantidades negativas, conteos no enteros y valores faltantes se rechazan con `422`.

## Errores

- `422`: entrada ausente o inválida; la respuesta omite los valores recibidos.
- `503`: modelo no disponible.
- `500`: fallo de predicción; se registra el detalle en el log del servidor, pero la respuesta no expone información interna.

## Pruebas

Con las dependencias instaladas, desde la raíz:

```powershell
.\.venv-1\Scripts\python.exe -m pytest 06_API_FASTAPI\tests -q
```

Las pruebas cubren rutas, documentación OpenAPI/Swagger, predicciones, validación, indisponibilidad del modelo y errores internos.

## Dependencias y seguridad

`requirements.txt` declara FastAPI, Pydantic, Uvicorn, cliente y runner de pruebas, además de las versiones del stack de ML con las que se generó el artefacto (Python 3.14.7, NumPy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1 y joblib 1.6.0). Cargar archivos Pickle/joblib puede ejecutar código: solo se deben usar modelos locales de confianza.