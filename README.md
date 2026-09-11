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

Para la RTX PRO 6000 Blackwell use Python 3.10--3.13 y cargue CUDA 12.8 o 13.0. El instalador detecta la versión de `nvcc`, crea `.venv`, instala PyTorch 2.10 desde el canal CUDA correspondiente y compila `tiny-cuda-nn` para `sm_120`:

```bash
# Si el clúster usa módulos, el nombre exacto puede variar.
module load cuda/12.8
./install_requirements.sh
```

Para usar otra versión de Python o ubicación del entorno:

```bash
PYTHON_VERSION=3.12 VENV_PATH="$PWD/venv" ./install_requirements.sh
```

No instale directamente el stack original PyTorch 1.12/CUDA 11.6: no contiene kernels para Blackwell. `torch-scatter` ya no es necesario; el proyecto usa la operación equivalente incluida en PyTorch.

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
