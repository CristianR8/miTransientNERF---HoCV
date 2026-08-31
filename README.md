# TransientNeRF: guía corta para estudiantes

Este repositorio implementa **Transient Neural Radiance Fields** (TransientNeRF), publicado en NeurIPS 2023. Aprende una representación 3D de una escena a partir de mediciones transitorias: cada píxel contiene intensidad en muchos instantes de tiempo, no solo un color.

Con una red entrenada se pueden generar transitorios, imágenes integradas y mapas de profundidad desde vistas de cámara no usadas para el entrenamiento.

## Estructura del proyecto

| Ruta | Para qué sirve |
| --- | --- |
| `train.ipynb` | Guía paso a paso para validar datos, crear la configuración y entrenar. |
| `train.py` | Ejecuta el ciclo de entrenamiento del método de los autores. |
| `eval.py` | Renderiza y calcula métricas con un modelo ya entrenado. |
| `utils.py` | Funciones compartidas: argumentos, renderizado y guardado de puntos de control. |
| `loaders/loader_synthetic.py` | Lee datos simulados. También lee directamente el archivo MITransient `scene_0.h5`. |
| `loaders/loader_captured.py` | Lee las escenas reales capturadas por los autores. |
| `loaders/utils.py` | Define los rayos de cámara usados por los cargadores. |
| `radiance_fields/` | Define la red neuronal que representa densidad y color de la escena. |
| `misc/transient_volrend.py` | Implementa el modelo de formación de imagen transitoria y la asignación a bins temporales. |
| `misc/summary.py` | Genera visualizaciones para TensorBoard durante el entrenamiento. |
| `misc/dataset_utils.py` | Funciones auxiliares para archivos HDF5 y datos. |
| `configs/train/` | Archivos `.ini` con parámetros de entrenamiento. |
| `configs/test/` | Archivos `.ini` con parámetros de evaluación. |
| `data/` | Datos locales. No se suben a Git. |
| `results/` | Modelos, imágenes y registros creados al entrenar. |

## Datos `scene_0.h5`

El archivo `data/scene_0.h5` contiene 25 vistas de una escena simulada:

- `images`: imágenes de estado estacionario.
- `poses`: matrices cámara-a-mundo de tamaño `4×4`.
- `transients`: mediciones transitorias de tamaño `256×256×1034×3` por vista.

El cargador usa experimentos de 2, 3 o 5 vistas para entrenar y reserva las demás vistas para comprobar la capacidad de síntesis.

## Instalación

El código original fue probado con Python 3.8--3.10 y CUDA 11.6. Cree un entorno virtual e instale las dependencias:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Si su versión de CUDA o Python es diferente, instale una versión compatible de PyTorch, torchvision, torchaudio, tiny-cuda-nn y torch-scatter antes de ejecutar el entrenamiento.

## Entrenamiento recomendado

1. Abra `train.ipynb` desde la raíz del repositorio.
2. Ejecute las celdas de validación.
3. Cambie `SCENE_AABB` por los límites reales de su escena, en metros.
4. Ejecute primero el entrenamiento corto de 100 pasos.
5. Si no hay errores, ejecute el entrenamiento completo.

También puede entrenar desde una terminal después de ejecutar la celda que crea la configuración:

```bash
python train.py -c configs/train/simulated/scene_0_5views.ini
```

El entrenamiento guarda puntos de control y registros dentro de `results/`. Para ver las curvas y las imágenes:

```bash
tensorboard --logdir results --port 6006
```

## Resultados esperados

Durante el entrenamiento se guardan:

- Pesos de la red y de la cuadrícula de ocupación (`*.pth`).
- Imágenes integradas a partir de los transitorios.
- Mapas de profundidad y opacidad.
- Gráficas de transitorios predichos y de referencia.
- Pérdida y PSNR en TensorBoard.

## Referencia

```bibtex
@inproceedings{malik2023transient,
  title = {Transient Neural Radiance Fields for Lidar View Synthesis and 3D Reconstruction},
  author = {Anagh Malik and Parsa Mirdehghan and Sotiris Nousias and Kiriakos N. Kutulakos and David B. Lindell},
  journal = {NeurIPS},
  year = {2023}
}
```
