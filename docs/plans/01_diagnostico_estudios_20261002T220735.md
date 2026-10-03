# Estudios STORM + RAINSTORM: diagnóstico y decisiones

Fecha: 2026-10-02, America/Argentina/Buenos_Aires. Análisis de código y lectura de la base local; no se modificaron planes ni se iniciaron entrenamientos.

## 1. Conclusión

Los fallos actuales tienen dos causas comprobadas: una preparación compartida incompatible con VAME oficial y un filtro de confianza que genera referencias coincidentes antes de orientar la pose. La selección del plan activo y la reconstrucción científica incompleta agregan problemas independientes. Resolver la excepción de orientación, por sí sola, no establece equivalencia con la tesis ni estabilidad de GPU.

La solución debe separar dataset, preparación, especificación del modelo y ejecución. Cada rama debe declarar su entrada y preparación. STORM debe validar y ejecutar esos contratos; RAINSTORM debe definir la ciencia, los formularios de pose y las recetas.

## 2. Evidencia de las corridas

Consulta de solo lectura de `.storm/studio.sqlite3`, tablas Job y Revision, estudio 8:

| Corrida | Revisión | Modelo | Resultado |
|---|---:|---|---|
| `9aedde53-3f88-4cbf-9a04-0cd382830d90` | 75 | vame_official | Rechaza pasos genéricos |
| `35f5cc72-c02a-4b1d-bd36-931821995ef5` | 74 | vame_native | Referencias de orientación coincidentes |
| `059e35a1-2142-45be-b3b8-cdeac2e14638` | 73 | identity | Referencias de orientación coincidentes |
| `f6072bed-17d8-400d-a078-f16085cf5a9a` | 72 | constant | Referencias de orientación coincidentes |

Todas derivan de la revisión 71, cuya rama principal es `online_mean`. Comparten el dataset crudo 11, sin receta de preparación guardada vinculada, y estos cinco pasos inline: selección de coordenadas, filtro de confianza, centrado, orientación y ventanas temporales. Las ramas reciben copias de esos pasos y de todas las métricas seleccionadas.

La rama nativa 74 usa `device:auto`, `window_storage:memory` y `rnn_backend:auto`. La revisión anterior 69 conserva `dataset_revision_id:12`, sin pasos adicionales, `device:cpu`, `window_storage:mapped` y `rnn_backend:native`. Las nuevas ejecuciones no usan ese plan. El dataset preparado existente no se recupera automáticamente al seleccionar el dataset crudo y volver a incluir pasos inline.

### 2.1 VAME oficial: incompatibilidad detectada demasiado tarde

`VAMEOfficialModel.supports_pipeline_steps=False`. El engine hace cumplir esa restricción, pero Studio comprueba pasos requeridos y configuración sin rechazar la combinación antes de encolar. `run_branches` cambia modelo/configuración y copia los mismos pasos. Por eso la ejecución 75 llega al worker con una configuración que el adapter expresamente prohíbe.

VAME oficial necesita archivos de pose de sesiones completas. Su adapter enlaza las sesiones de entrenamiento registradas y llama a `vame.preprocessing`, `create_trainset`, `train_model` y `segment_session`. Enviar ventanas del pipeline nativo o simplemente usar un dataset preparado no resuelve este contrato. Los recortes, reservas parciales o discontinuidades requieren una estrategia explícita de fuentes segmentadas que el adapter actual no ofrece de forma general.

### 2.2 Orientación: el filtro introduce el problema

Prueba de lectura dentro del worker, usando el H5 registrado de `Hab1_2025-08-21T15_11_04` y las clases reales `LikelihoodFilter` y `OrientPose`, en CPU:

| Comprobación | Resultado |
|---|---:|
| Frames de la sesión | 18.636 |
| Nose y body coincidentes en el H5 original | 0 |
| Frames con ambos puntos por debajo de confianza 0,6 | 75 |
| Frames coincidentes después del filtro actual | 62 |
| Primera coincidencia producida | Frame 0 |

El filtro usa el último punto confiable solo si su frame es inmediatamente anterior. Si no existe, o la baja confianza dura varios frames, escribe `(0,0)`. Cuando ambos puntos quedan en cero, la orientación falla con `degenerate_reference_policy:error`, valor que usa el resolver cuando la receta lo omite. Centrar no elimina la coincidencia.

La prueba reproduce `ValueError: Orientation reference points coincide`. Confirma el mecanismo en una sesión; no es un censo de las 86 sesiones. Identity y constant fallan antes de ejecutar sus modelos porque heredan la misma preparación.

Elegir `identity` como política de orientación puede evitar la excepción, pero deja frames sin orientar. Debe ser una variante explícita, con máscara y conteos; no sustituye la receta histórica.

### 2.3 El plan mostrado puede cambiar al ejecutar

Studio obtiene el plan con `kind='plan'` y mayor PK. Las variantes creadas por `run_branches` también tienen `kind='plan'`. La última rama pasa a ser el plan mostrado por la UI. En este caso la última revisión es 75, VAME oficial, aunque la raíz elegida era 71.

Además, las variantes conservan `branch_models` y `branch_configs`, de modo que abrir o volver a ejecutar una variante puede presentar otra vez ramas heredadas. Es necesario distinguir una revisión editable del experimento de una especificación inmutable de ejecución.

## 3. Diferencias con Tesis_Facu

| Aspecto | Tesis_Facu | Integración actual | Consecuencia |
|---|---|---|---|
| Confianza | Ingesta marca coordenadas inválidas con NaN | Filtro reemplaza por punto previo inmediato o cero | Cambia los datos y puede crear referencias coincidentes |
| Limpieza de pose | Filtrado por velocidad y geometría; después interpolación PCHIP, relleno de bordes, mediana de 5 frames y Gaussian smoothing | Preset nativo omite filtrado anatómico/velocidad y suavizado | La entrada reconstruida no equivale a la histórica |
| Unidades y ejes | Ingesta ofrece px/cm y conversión de eje Y; transformaciones operan sobre esa representación | Cargador DLC recupera XY y metadatos; no reproduce explícitamente toda esa secuencia histórica | Deben verificarse configuración real, unidades y orientación antes de portar umbrales |
| Orientación degenerada | `SpatialAlignment` calcula `arctan2` sin la excepción explícita actual | `OrientPose` aborta cuando el vector es cero | El nuevo criterio estricto necesita diagnóstico y política versionada |
| ROI | Builder incorpora ingesta/augmentation, segmentos y métricas; centrado/alineación pueden transformar ROI | Preset nativo usa 30 XY; preset oficial describe ROI como contexto posterior | Tener ROIs registrados no implica haber generado las características/contexto de la tesis |
| VAME nativo | Red configurable, cabeza predictiva opcional, reducer/clusterer componibles; trainer ofrece clipping, warmup KL y validación | Red GRU con reconstrucción variacional y KMeans fijo; no expone esas opciones del trainer | Comparten una familia de arquitectura, no toda la implementación configurable |
| VAME oficial | Helpers y recetas finales dedicados al backend oficial | Adapter usa el backend real, pero está incluido en ramas con preparación nativa | Error de composición del estudio, anterior al entrenamiento oficial |
| Receta oficial final | Referencia `ego_roi`: KMeans, learning rate 0,0005 | Rama 75 usa defaults HMM y no registra ese learning rate en config_kwargs | Seleccionar el nombre del modelo no carga necesariamente su receta final |
| Métricas | Evaluaciones dedicadas a cada tarea | Plan 71 incluye a la vez clasificación, clustering y regresión | Configuración aceptada no demuestra que todas las métricas sean interpretables |

La arquitectura básica de reconstrucción nativa se parece a la versión configurable de la tesis. La existencia de opciones allí no prueba que todas se usaran en el entrenamiento final: recuperar la configuración concreta forma parte del backlog. Tampoco debe añadirse normalización o warmup por suposición.

El preset registrado ya declara que su reconstrucción es parcial. `recipe-reference.json` declara `historical_reference_not_retraining_recipe`. Esto debe verse al elegir la receta, con las etapas pendientes, y no quedar relegado a documentación.

## 4. Límites entre repositorios

### STORM: contratos y ejecución generales

- Identidad de fuentes, observaciones, sesiones/segmentos, particiones, reservas y revisiones.
- Registro versionado de conectores, procesadores, modelos, métricas y visualizaciones.
- Esquemas de entradas/salidas y capacidades: preparación externa o interna, operaciones admitidas, granularidad requerida, alcance de estados, checkpoint/reanudación y entorno.
- Validación compartida entre API Python, UI y worker; especificaciones serializables independientes de Django.
- Editor gráfico general, formularios derivados de esquemas, historial, artefactos, ejecución persistente, eventos, diagnóstico de comparación y exportación.

### RAINSTORM: implementación científica y experiencia de pose

- DLC/H5, ROI, video, taxonomías, conversiones de unidades/ejes y características.
- Políticas de pose inválida, interpolación, limpieza, centrado, orientación y ventanas.
- Recetas y modelos VAME nativo/oficial, supervisados y sus entornos.
- Resolución de nombres de partes a columnas; controles y previews de pose/ROI.
- Compiladores de arquitectura y validadores científicos específicos.

Actualmente STORM Studio contiene `resolve_preparation_steps` con reglas `pose.*`, `PREPARATION_STEP_UI`, previews de pose y mensajes especiales de VAME. Mover ese conocimiento a extensiones de RAINSTORM por etapas. Conservar en Studio los hosts genéricos y compatibilidad con recetas previas; no reescribir todo el engine.

## 5. Extensión gráfica y en código

Sí es viable ampliar ambos caminos sobre una misma especificación versionada:

1. **Componer componentes existentes:** arrastrar nodos, configurar parámetros y conectar puertos tipados. Primera entrega; no requiere programar.
2. **Crear variantes de modelos:** guardar receta/arquitectura con bloques admitidos por el backend. RAINSTORM compila bloques concretos y valida shapes, pérdidas y salida. VAME oficial se controla mediante las opciones reales de su adapter.
3. **Agregar componentes nuevos por código:** plantilla de plugin, esquema, entrada/salida, operaciones, pruebas y runtime. Registrar el paquete y que aparezca en el mismo catálogo gráfico. Un editor de código opcional puede publicar una versión del plugin en un worker aislado; no ejecutar Python arbitrario dentro de Django.

Reflection ayuda a descubrir firmas, métodos y esquemas existentes. No puede deducir significado de la clase positiva, fuga de datos, equivalencia de recetas, soporte de reanudación ni granularidad científica. Esas propiedades deben declararse y probarse.

Toda variante conserva manifest, código/paquete y hash, entorno, receta, shapes y procedencia. Importar pesos exige contrato de entrada y significado de salida. Mostrar las operaciones admitidas por cada adapter; no ofrecer un editor libre de capas para un backend que no lo implementa.

## 6. Flujo de UI propuesto

**Datos → Preparación → Experimento → Ejecuciones → Resultados.**

- **Datos:** elegir fuentes/adapters; inventario por sesión, particiones y benchmark protegido. Dataset siempre independiente del modelo.
- **Preparación:** canvas de transformaciones; entrada/salida de cada nodo, preview antes/después sobre los mismos frames, unidades, confianza, exclusiones y artefactos reutilizables.
- **Experimento:** una tarjeta por rama con modelo, versión, dataset/receta de entrada, configuración, entorno y métricas válidas. “Comparar modelos” comparte la cohorte de evaluación y alineación; cada rama conserva su preparación.
- **Validar:** informe de errores por rama, paso y campo, con enlace al control correspondiente. Ejemplo: “VAME oficial prepara sus H5 internamente; esta rama tiene 5 transformaciones externas”. No crear jobs si la validación falla.
- **Ejecuciones:** fase/subfase, sesión o lote, avance medido, actividad del worker separada del avance, ETA cuando haya datos, logs expandibles, error y último artefacto recuperable.
- **Resultados:** cobertura y exclusiones, distribución de estados, duraciones, matriz de transiciones, métricas admitidas y video/pose sincronizados. Cada gráfico permite consultar tabla y descargar datos. Desacuerdo entre IDs de motivos requiere métricas invariantes a permutaciones.

## 7. Verificación realizada y límites

Se leyeron los planes de las cuatro corridas y su raíz en la base real, se contrastó el flujo de ramas con el código y se reprodujo el fallo con clases reales y una sesión registrada. No se ejecutó entrenamiento completo ni se comprobó equivalencia numérica completa con la tesis. No se repitió la suite de tests al generar estos documentos.

La aceptación documentada todavía deja abiertos entrenamiento biológico nativo completo, reconstrucción oficial final y comparación contra el benchmark. Los GPU Hang anteriores requieren un diagnóstico de runtime independiente; una corrección de preparación no prueba que estén resueltos.

## Fuentes locales revisadas

- STORM: `packages/storm-studio/src/storm_studio/views.py`, `forms.py`, `services.py`, `data_preparation.py`; `packages/storm-engine/src/storm/suite.py`.
- RAINSTORM: `packages/rainstorm-thesis/src/rainstorm_thesis/{plugin,preprocessing,data,vame_models,vame_native_runtime,vame_official_runtime}.py`; `docs/reconstruction/{recipe-reference.json,acceptance-status.md}`.
- Tesis_Facu: `src/rainstorm/experiments/study/{input_builder,specs}.py`; `domains/pose/preprocessing/{tracking_ingestion,alignment,filtration,refinement}.py`; `application/steps/preprocessing/{pose_filtration_step,pose_smoothing_step}.py`; `experiments/models/vame/{vame_config,vame_neural_model,vame_trainer,vame_loss,vame_model}.py`; `experiments/models/commons/decoder_gru.py`.

El backlog y los criterios de aceptación están en [02_plan_backlog_estudios_20261002T220735.md](02_plan_backlog_estudios_20261002T220735.md).

## Seguimiento posterior a este diagnóstico

Este registro fechado no describe estado en vivo. [Backlog definitivo](02_plan_backlog_estudios_20261002T220735.md), [épica RAINSTORM](https://github.com/munhof/RAINSTORM/issues/1) y [épica STORM](https://github.com/munhof/STORM/issues/1).
Las revisiones/fallos históricos se preservan; contratos implementados no equivalen a reconstrucción científica.
