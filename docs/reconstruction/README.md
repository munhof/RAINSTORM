# Estado de la reconstrucción RAINSTORM + STORM

RAINSTORM aporta los conectores científicos y modelos; STORM Studio registra
las fuentes, versiones de preparación, ejecuciones y comparaciones. Los datos
se registran aparte de los modelos. Los hashes identifican los archivos de
origen sin copiarlos al manifiesto.

## Datos y preparación

El inventario histórico de `tests/fixtures/historical_inventory.json` conserva
los 86 nombres observados en el notebook
`Tesis_Facu/notebooks/model_experiments/00_dataset_inventory_split.ipynb`.
La receta reproduce el seed 156, el orden por nombre y los grupos de 10
sesiones exploratorias, 62 de entrenamiento, 22 de test y 2 de validación
interna. El notebook muestra listas completas para exploración, test y
validación; su salida trunca la lista de entrenamiento, así que allí sólo se
puede comprobar la pertenencia de las sesiones visibles.

En Studio se pueden registrar poses DLC H5/CSV, videos, ROI y CSV de etiquetas.
Videos, ROI y etiquetas se vinculan explícitamente a sesiones de pose. Guardar
un cambio crea una nueva revisión del dataset; las fuentes, vínculos y
resultados de inventario anteriores se conservan. Las sesiones nuevas que no
estén en las particiones quedan como `sin asignar` hasta que el investigador
las ubique.

La pipeline de datos se arma con pasos codeless y versionados, independientes
del modelo: selección de coordenadas, centrado, orientación, filtro de
likelihood y ventanas temporales. El inventario muestra metadatos, particiones,
segmentos, features y una muestra de hasta 256 observaciones. Para recorrer la
sesión completa, el worker guarda coordenadas en bloques comprimidos y Studio
solicita sólo el bloque visible. La vista sincroniza el video usando el mapeo de
frames del adapter y superpone las geometrías ROI compatibles. El calibrador
permite elegir cualquier observación de pose, ubicar visualmente el mismo
instante en el video y cargar el offset calculado en el vínculo. Guardar los
vínculos crea una revisión nueva. Las revisiones inventariadas antes de esta
función deben actualizarse para crear su índice de navegación.

Si el conector aporta targets, la misma vista muestra la etiqueta de referencia
del frame, su taxonomía y si pasa la máscara de etiqueta válida. Los frames sin
etiqueta válida quedan identificados explícitamente y no se convierten en una
clase por defecto.

En el mismo timeline se pueden seleccionar hasta cuatro corridas completadas
del mismo origen para ver sus salidas sobre el frame actual. Studio las alinea
por ID de observación y comprueba sesión y frame; una corrida de otro dataset o
sin esa correspondencia no aparece como opción. Los valores conservan la
semántica declarada por cada adapter, incluidos los IDs locales de motivos.
Las anotaciones versionadas de una corrida también aparecen en una capa humana
separada: los intervalos siguen el índice de observación original y las
interpretaciones de motivos sólo se muestran si el fingerprint coincide.

La vista también permite guardar una corrección para el frame seleccionado,
con autor y motivo. Cada envío crea una revisión inmutable enlazada con la huella
del dataset; conserva tanto la etiqueta original como la corrección y no modifica
los archivos fuente. La corrección sigue separada del dataset. Para usarla, el
investigador debe seleccionarla explícitamente en una nueva revisión del plan;
el worker aplica sólo las etiquetas de observaciones asignadas a entrenamiento
y fuera de toda reserva de evaluación. Las demás quedan registradas como
excluidas, y el reporte conserva el ID de la revisión de correcciones.
Los filtros de frames con referencia válida, corrección o anotación permiten
avanzar o retroceder por el timeline leyendo bloques. Un snapshot guarda el
filtro activo, las corridas seleccionadas, la sesión, el punto corporal y el
frame. La vista recupera esos valores al restaurar el snapshot.

En Reportes se puede congelar una selección en HTML, CSV y JSON. El manifiesto
guarda resultados, planes, revisiones de datos, fuentes con hash, anotaciones y
el snapshot visual elegido. Las anotaciones con otro fingerprint permanecen
como historial y no se proyectan sobre las predicciones. El enlace **Paquete ZIP**
incluye las fuentes, artefactos, checkpoints disponibles y vistas de pose
vinculados al reporte; verifica los hashes registrados y copia los binarios por
bloques. Los payloads pickle se transportan como archivos opacos, sin
deserializarlos. El ZIP conserva evidencia, pero todavía no restaura
automáticamente el estudio en otro workspace.

Si el navegador no decodifica el formato o codec fuente, Datos ofrece preparar
una copia H.264/MP4 como trabajo del worker. La copia usa como máximo 960 píxeles
de ancho, mantiene el frame rate y las posiciones de frame, se publica de forma
atómica y conserva intacta la fuente. La vista sigue admitiendo solicitudes HTTP
por rango para que el reproductor pueda buscar. En el video reconstruido de
Manual Labels se verificaron 33.000 frames, 30 FPS y 1.100 segundos; el AVI de
1.576.780.920 bytes produjo una copia de 96.760.481 bytes.

El CSV de `Manual Labels` usa `Frame` con origen 1; el índice H5 y el frame de
video usan origen 0. El benchmark conserva esa conversión, el recorte inclusivo
1–7.200 y una máscara de exclusión: las filas ausentes, inválidas o ambiguas no
se convierten en una clase. Para la fuente reconstruida hay 33.000 frames de
pose/video y 5.303 etiquetas válidas dentro del recorte. El inventario compara
los timestamps del video con el FPS declarado: `Manual Labels` quedó verificado
como continuo. Si encuentra un salto, lo muestra en Datos y lo convierte en un
límite de segmento para que la preparación y los modelos temporales no creen
ventanas que lo crucen. Datos permite reemplazar esos límites por sesión y crear
una revisión nueva del dataset; la detección original queda registrada y el
worker aplica los límites seleccionados al segmentar las observaciones. La nueva
revisión vuelve a inspeccionarse y requiere una decisión humana propia. Si no
puede verificar los timestamps, marca la continuidad como desconocida y exige
que el investigador confirme su revisión. La decisión guarda autor, motivo y la
huella de los videos, asociaciones y límites revisados. Preparación, vista previa
y ejecución de modelos quedan bloqueadas hasta guardarla; cambios en los límites
requieren otra revisión.

## Benchmark NOR de ejemplo

En **Datos → Benchmarks de ejemplo**, Studio puede registrar `NOR_TS_01` a
`NOR_TS_10` como una revisión de evaluación separada: 10 poses DLC H5, 10 videos
de 25 FPS, `ROIs.json`, 10 CSV manuales y `reference.json`. Son 75.000 filas;
7.427 tienen una etiqueta one-hot válida. El resto queda en la máscara de
exclusión y no participa de las métricas. La correspondencia entre `obj_1` /
`obj_2` y `Known` / `Novel` se toma de `reference.json` por sesión. CSV empieza
en frame 1; pose y video, en frame 0.

La revisión queda marcada como `inference_only`, con todas las sesiones en
evaluación y todos sus frames protegidos. Studio impide entrenar, actualizar,
reanudar o reprocesar con ella. Después de inspeccionarla, se puede aplicar una
corrida guardada desde **Modelos**; los adapters de inferencia preentrenados se
pueden configurar desde **Flujo**. La pose alimenta los adapters; los videos y
ROI quedan asociados para la revisión visual y contextual.

Para VAME, el reporte calcula ARI, NMI, homogeneidad y completitud contra las
etiquetas `Known` / `Novel`; estos puntajes no comparan directamente IDs de
motivos. Para salidas binarias calcula accuracy, precision, recall, F1 y
balanced accuracy sólo cuando la correspondencia declarada por el adapter
coincide con la taxonomía. La semántica de los bundles `example_simple` y
`example_wide` sigue desconocida, así que no se los presenta como comparables
hasta que el investigador confirme esa correspondencia.

## Modelos y comparaciones

La [procedencia supervisada recuperada](supervised-provenance.md) distingue
la receta histórica de exploración del origen aún desconocido de los pesos.

El registro de Studio incluye cuatro adapters:

- `vame_native` entrena el VAE GRU de RAINSTORM y agrupa el espacio latente con
  KMeans. Además de las pruebas sintéticas, pasó un smoke biológico con la
  preparación `pose_ego`: se entrenó una época con 1.024 frames de dos H5,
  generó 1.012 ventanas y, tras guardar y recargar con `FileArtifactStore`,
  infirió 506 ventanas de una tercera sesión de 512 frames. El smoke aplicó
  filtro de confianza, centrado, orientación y ventanas temporales; para los
  frames donde coinciden las referencias usó la opción explícita de conservar
  el frame sin rotación. Guarda checkpoints al final de cada época con pesos,
  optimizador y estados aleatorios; una ejecución interrumpida puede continuar
  desde la última época compatible. El entrenamiento final con las 62 sesiones
  aún no está verificado.
- `vame_official` ejecuta las etapas de `vame-py` para preparar, entrenar y
  segmentar sesiones completas. Al aplicar un modelo guardado a nuevas sesiones
  registradas, recupera el encoder y usa el HMM o los centros KMeans guardados;
  no vuelve a ajustar el discretizador con los datos de inferencia. Además de
  los fixtures, pasó un smoke biológico multisesión: entrenó durante una época
  con dos sesiones H5 reales (6.552 y 6.915 frames) y aplicó encoder/KMeans
  guardados a una tercera sesión de 14.580 frames. El adapter entrenado se guardó
  con `FileArtifactStore`, se recargó y volvió a inferir con 14.562 ventanas
  válidas, conservando el discretizador del modelo. La receta histórica completa
  (50 épocas, 50 estados) y el entrenamiento final con las 62 sesiones aún no
  están verificados.
- `supervised_simple` y `supervised_wide` cargan los bundles Keras incluidos,
  en un runtime separado de Studio. Sus entradas son `(N, 12)` y `(N, 7, 12)`;
  producen una probabilidad binaria. No se conoce su población de entrenamiento
  y no se ofrece reentrenamiento. Ambos se probaron mediante inferencia STORM
  con los pesos reales y luego con el artefacto de modelo recargado. Es un smoke
  de ejecución, no una evaluación de precisión ni de validez científica.

Las cifras y parámetros del manuscrito son referencias históricas, no resultados
de una reconstrucción nueva. `docs/reconstruction/recipe-reference.json`
registra esa distinción.

La comparación comprueba datos, observaciones, máscara, tarea, taxonomía y
métricas. Un modelo binario sólo puede entrar en un ranking común si se declara
una correspondencia versionada y compatible con una taxonomía de dos clases;
la salida binaria no se equipara automáticamente a las 12 categorías humanas.
Los IDs de motivos VAME son categorías locales al modelo y no se interpretan
como conductas compartidas entre corridas.

## Pendiente para la aceptación completa

- Ejecutar la reconstrucción final de VAME oficial con las 62 sesiones de train;
  el smoke biológico de dos sesiones y la inferencia tras recargar el artefacto
  ya están verificados.
- VAME oficial requiere sesiones completas, continuas y sin frames reservados.
- Implementar un importador del ZIP que reconstruya el estudio en otro workspace;
  la exportación de evidencia ya está disponible.
- Completar la revisión de teclado y viewport reducido del timeline. Preparación,
  selección de comparación y tablas de reportes ya tienen pruebas de navegador
  a 390 píxeles.

## Entornos de ejecución

TensorFlow/Keras 2.10.1 corre en Python 3.9, separado de Studio. El worker
científico de VAME fija Python 3.12 y sus dependencias en
`envs/vame_worker/uv.lock`. Studio debe invocar el intérprete exacto configurado
para cada runtime, sin cargar TensorFlow o PyTorch en el proceso web. Las
instrucciones para registrar el plugin y levantar el worker están en
[`worker-adapters.md`](worker-adapters.md).

Para repetir el smoke biológico de VAME nativo con los H5 locales, definí la
ruta que contiene `rainstorm_analysis_simon` y ejecutá:

```bash
RAINSTORM_NATIVE_VAME_BIOLOGICAL_SMOKE=1 \
RAINSTORM_RECONSTRUCTION_ROOT=/ruta/a/los/datos \
uv run --project envs/vame_worker --locked python -m pytest -q \
  tests/test_reconstruction.py::test_native_vame_pose_ego_trains_and_reloads_on_biological_h5_sessions
```
