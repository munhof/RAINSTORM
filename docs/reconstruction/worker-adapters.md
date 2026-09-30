# Worker RAINSTORM: carga, preparación y modelos

RAINSTORM registra en STORM los conectores `dlc_h5` y `dlc_csv`, los pasos de
preparación de pose, los modelos `vame_native`, `vame_official`,
`supervised_simple` y `supervised_wide`, y las visualizaciones de pose y
estados. El plugin carga las dependencias de lectura y ML al ejecutar el
trabajo, no al abrir Studio.

## Arrancar Studio y el worker con Podman Compose

Desde la raíz del checkout de STORM, aplicá el overlay de RAINSTORM. El compose
construye Studio y el worker científico, monta el plugin y conserva ambos
servicios en segundo plano:

```bash
podman-compose -f compose.yaml -f ../RAINSTORM/compose.storm-plugin.yaml up -d --build
```

Studio queda en <http://127.0.0.1:8000>. Para comprobar servicios y logs:

```bash
podman-compose -f compose.yaml -f ../RAINSTORM/compose.storm-plugin.yaml ps
podman-compose -f compose.yaml -f ../RAINSTORM/compose.storm-plugin.yaml logs -f worker
```

## Ejecutar el worker fuera de Compose

Como alternativa, podés detener el worker del compose y ejecutarlo desde su
entorno local. El entorno fija Python 3.12 y las dependencias de VAME; Studio
conserva su entorno liviano:

```bash
cd /ruta/al/STORM-System-for-Traceable-Orchestration-Reuse-and-Modeling
podman-compose -f compose.yaml -f ../RAINSTORM/compose.storm-plugin.yaml stop worker
cd /ruta/al/RAINSTORM
uv sync --project envs/vame_worker --locked
STORM_WORKSPACE=../STORM-System-for-Traceable-Orchestration-Reuse-and-Modeling/.storm \
STORM_PLUGINS=rainstorm_thesis.plugin \
uv run --project envs/vame_worker storm-studio worker --recover-stale
```

El lock usa PyTorch CPU. VAME oficial guarda su proyecto bajo
`STORM_WORKSPACE/rainstorm/vame_official`, compartido con Studio. El worker
comparte la base de datos y los artefactos. Para volver al worker del compose,
detené este proceso con `Ctrl+C` e iniciá de nuevo el overlay de Podman Compose.

## Worker CUDA en una VM con Podman

La variante CUDA conserva el worker CPU y usa su propio lock, imagen y overlay.
En la VM instalá Podman, `podman-compose` con soporte para reservas GPU, el
driver NVIDIA y NVIDIA Container Toolkit con CDI. La [guía de CDI de
NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)
describe la integración con Podman; `nvidia-ctk cdi list` debe mostrar la L4.
El lock CUDA instala PyTorch `2.14.0+cu130`; el driver y la exposición del
dispositivo corresponden al host. Como primera VM, `g2-standard-8` ofrece 8
vCPU, 32 GB de RAM y una L4 de 24 GB ([tipos GPU de Compute
Engine](https://docs.cloud.google.com/compute/docs/gpus)).

Desde la raíz del checkout de STORM, levantá Studio y el worker CUDA:

```bash
podman-compose \
  -f compose.yaml \
  -f ../RAINSTORM/compose.storm-plugin.yaml \
  -f ../RAINSTORM/compose.storm-plugin.cuda.yaml \
  up -d --build
```

En la configuración del modelo `vame_native`, seleccioná `device: cuda`; el
valor inicial del formulario sigue siendo `cpu`. El overlay entrega la GPU sólo
al contenedor `worker`.

El compose publica Studio únicamente en `127.0.0.1:8000`. Para abrirlo desde tu
PC sin IP pública ni entrada directa al puerto 8000, permití SSH desde IAP en el
firewall de la VM. Asigná `roles/iap.tunnelResourceAccessor` sobre esa VM y el
permiso SSH correspondiente; si activás OS Login, usá `roles/compute.osLogin`.
Google documenta la [regla de firewall y los permisos de IAP TCP
forwarding](https://docs.cloud.google.com/iap/docs/using-tcp-forwarding). En tu
PC, abrí el túnel:

```bash
gcloud compute ssh NOMBRE_VM --zone=ZONA --tunnel-through-iap -- \
  -N -L 8000:127.0.0.1:8000
```

Después abrí <http://127.0.0.1:8000>. El túnel requiere la regla de firewall
SSH desde `35.235.240.0/20`; no requiere una regla pública para el puerto 8000.

## Worker ROCm en Arch Linux con Podman

El host ya cuenta con `rocm-core`, `hip-runtime-amd` y `rocminfo` del repositorio
oficial `Extra`; no hace falta reemplazarlos por paquetes AUR. El perfil ROCm
mantiene PyTorch aislado dentro de un worker basado en la imagen AMD para ROCm
10.0, con kernels `gfx1103` para la Radeon 780M. La imagen y el lock son
independientes del worker CPU.

Antes de levantarlo, comprobá que `rocminfo` enumere `gfx1103` y que existan
`/dev/kfd` y `/dev/dri/renderD*`. En una instalación rootless, el overlay pasa
los grupos suplementarios del host al worker (`keep-groups`); si el acceso a
los dispositivos está restringido, agregá tu usuario a `render` y `video` y
volvé a iniciar sesión. No hace falta instalar PyTorch ROCm en el Python del
host.

Desde la raíz del checkout de STORM, levantá Studio y el worker ROCm:

```bash
podman-compose \
  -f compose.yaml \
  -f ../RAINSTORM/compose.storm-plugin.yaml \
  -f ../RAINSTORM/compose.storm-plugin.rocm.yaml \
  up -d --build
```

La primera construcción descarga la imagen de AMD. Verificá que PyTorch vea el
backend HIP y la GPU:

```bash
podman-compose \
  -f compose.yaml \
  -f ../RAINSTORM/compose.storm-plugin.yaml \
  -f ../RAINSTORM/compose.storm-plugin.rocm.yaml \
  exec worker python -c 'import torch; print(torch.__version__, torch.version.hip); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU ROCm no disponible"); assert torch.version.hip and torch.cuda.is_available()'
```

En los planes de `vame_native`, configurá `device: cuda`: PyTorch usa esa API
también para HIP/ROCm. La prueba del dispositivo debe pasar antes de iniciar el
entrenamiento. La receta fija PyTorch `2.13.0+rocm10.0.0`, Triton y los paquetes
de `gfx1103` publicados por AMD.

## Registrar y revisar datos

En la revisión de Datos podés registrar archivos H5/CSV DLC, videos, JSON de
ROI y CSV de anotaciones. Vinculá cada video, ROI o CSV con una sesión de pose;
las asociaciones se guardan como una revisión nueva del dataset. Si agregás
pose después de configurar las particiones, las sesiones existentes conservan
su partición y las sesiones nuevas quedan como `sin asignar`.

El inventario informa sesiones, frames, features, segmentos y etiquetas, con
una tabla de hasta 256 observaciones representativas, distribuidas por sesión y segmento. La vista de pose pagina bloques comprimidos para recorrer todos los frames de cada sesión sin incrustar el dataset en la página. Muestra los ROI compatibles con la geometría declarada y sincroniza el video mediante el mapeo de frames del adapter. El calibrador permite elegir cualquier frame de pose, ubicar visualmente el instante correspondiente en el video y cargar el offset al vínculo. Guardar el vínculo crea una revisión nueva; el offset sigue requiriendo una correspondencia visual del investigador.

## Armar una pipeline codeless

Los pasos de preparación se configuran por separado del adapter de origen y
del modelo. Studio conserva los parámetros, el orden, las entradas y la
revisión resultante. El conector recibe `pose_path`/`pose_paths`, los IDs
`pose_session_ids`, `video_path`/`video_paths`, etiquetas por sesión, FPS,
bodyparts, particiones y el recorte `[frame_start, frame_stop)`. Calcula hashes
de las fuentes. Las etiquetas usan `Frame` de base 1 y los índices H5/video de
base 0 por defecto; se pueden declarar otros orígenes.

Pasos disponibles:

- `pose.select_coordinates`: seleccionar columnas por nombre o índice.
- `pose.recenter`: recentrar pares x/y respecto de un punto corporal.
- `pose.orient_coordinates`: rotar coordenadas según dos referencias. En la
  UI elegís qué hacer si coinciden: detener y revisar (predeterminado) o dejar
  ese frame sin rotación. La segunda opción conserva las coordenadas ya
  centradas y queda guardada en la receta.
- `pose.likelihood_filter`: reemplazar coordenadas con baja confianza usando
  el último valor válido del mismo segmento.
- `pose.temporal_windows`: formar ventanas sin cruzar sesiones, segmentos,
  particiones ni frames reservados.

## Configurar modelos

`vame_native` entrena el VAE GRU de RAINSTORM y agrupa el espacio latente con
KMeans. Su receta inicial usa 50 estados, dimensión latente 20, 25 épocas,
batch 256, KLD 0.5 y seed 156. El adapter permite inferencia desde el estado
del entrenamiento guardado. También guarda checkpoints al final de cada época
con pesos, optimizador, historial y estados aleatorios para reanudar un trabajo
interrumpido. La continuación valida datos, receta y versiones de PyTorch y
scikit-learn. Los IDs son categorías locales a cada modelo.

`vame_official` llama a las etapas de `vame-py` para crear el proyecto,
preprocesar, entrenar y segmentar. Acepta sesiones completas de entrenamiento,
continuas y sin frames reservados; no acepta recortes o ventanas parciales. El
presupuesto `max_epochs` debe ser al menos 2: el bucle oficial empieza en la
época 1 y con valor 1 no ejecuta entrenamiento. El
adapter puede aplicar un encoder guardado a sesiones completas registradas y
reutiliza el discretizador ajustado en entrenamiento. VAME oficial requiere
sus archivos de pose y metadatos de sesión; la caja de inferencia JSON queda
deshabilitada. Este adapter realiza el preprocesado oficial internamente y
bloquea pasos codeless genéricos para que el plan no informe transformaciones
que el backend ignoraría. Un smoke biológico entrenó dos sesiones por una época
y aplicó el encoder y discretizador guardados a una tercera; la receta histórica
completa de 50 épocas y el entrenamiento con las 62 sesiones todavía no están verificados. Los ROI se registran
y se muestran en Studio, pero hoy no se pasan como entrada contextual al adapter.

`supervised_simple` y `supervised_wide` cargan los bundles Keras existentes en
un runtime TensorFlow aislado. Reciben respectivamente 12 features y ventanas
de 7 × 12, y producen una probabilidad binaria. El runtime informa que la
población de entrenamiento es desconocida; no ofrece `fit`.

Los parámetros del manuscrito son referencias históricas para ajustar la
configuración. Las ejecuciones nuevas deben conservar su propia revisión,
receta, modelo y resultados. Para comparar una salida binaria con etiquetas
humanas, primero declarás una correspondencia versionada con la taxonomía de
dos categorías. Los IDs de motivos VAME no se comparan como etiquetas
semánticas compartidas entre corridas.

## Límites de recuperación y evidencia

- Los pasos de preparación e inferencia guardan la ejecución en STORM; cerrar
  el navegador no cancela un worker activo.
- VAME nativo puede continuar desde límites de época; no retoma batches
  intermedios. VAME oficial todavía no declara recuperación de entrenamiento.
- VAME oficial requiere fuentes continuas; registrá y revisá los segmentos
  antes de entrenar para evitar que sus ventanas crucen discontinuidades.
- El timeline de Studio muestra los estados como categorías y no calcula por sí
  solo métricas de agrupamiento o equivalencias entre modelos.
- Las revisiones y asociaciones de datos se versionan. El bloqueo independiente
  de correcciones aceptadas y la restauración automática desde un ZIP exportado
  siguen pendientes. El ZIP de evidencia y sus hashes ya se pueden descargar.
