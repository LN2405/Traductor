# Traductor LSP

Proyecto local para capturar datos con MediaPipe, entrenar un modelo LSTM y detectar señas en tiempo real.

## Requisitos

- Python 3.10 recomendado
- Camara web
- Windows PowerShell

## Crear entorno virtual

Desde la carpeta del proyecto:

```powershell
python -3.10 -m venv venv
```

Activar el entorno:

```powershell
.\venv\Scripts\Activate
```

Si PowerShell bloquea la activacion, ejecuta:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Luego vuelve a activar:

```powershell
.\venv\Scripts\Activate
```

## Instalar dependencias

Con el entorno activado:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Comandos principales

Grabar datos para una letra o palabra:

```powershell
python -m training.collect
```

Entrenar el modelo:

```powershell
python -m training.train
```

Ver diagnostico del modelo:

```powershell
python -m training.diagnostics
```

Sincronizar vocabulario con la API:

```powershell
python -m training.api_sync
```

Ejecutar deteccion en vivo:

```powershell
python detect_local.py
```

Para cerrar la ventana de deteccion o grabacion, presiona:

```text
q
```

## Configuracion

Los valores principales estan en:

```text
app/config.py
```

Actualmente:

```python
NO_SEQUENCES = 30
SEQUENCE_LENGTH = 30
```

- `NO_SEQUENCES`: cantidad de videos/secuencias nuevas que se graban por clase.
- `SEQUENCE_LENGTH`: cantidad de frames por secuencia. El modelo espera 30.

## Archivos importantes

```text
detect_local.py                 Deteccion en vivo
training/collect.py             Grabacion de datos
training/train.py               Entrenamiento
training/diagnostics.py         Matriz de confusion y diagnostico
training/api_sync.py            Sincronizacion de vocabulario
app/services/mediapipe_service.py Extraccion de landmarks
app/model/action_lsp.h5         Modelo entrenado
app/model/labels.json           Labels del modelo
data/MP_Data                    Dataset local
```
