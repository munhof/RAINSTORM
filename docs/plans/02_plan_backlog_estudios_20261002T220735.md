# Plan y backlog: estudios científicos y extensión low code

## Actualización vigente: contratos y tickets definitivos

Fecha: 2026-10-02. La propuesta original se conserva más abajo como evidencia
histórica; este seguimiento y los issues definitivos rigen la primera entrega.
Épicas relacionadas: [RAINSTORM](https://github.com/munhof/RAINSTORM/issues/1) y [STORM](https://github.com/munhof/STORM/issues/1).

[Diagnóstico fechado](01_diagnostico_estudios_20261002T220735.md).

| ID | Prioridad | Issue definitivo | Estado de esta entrega |
|---|---|---|---|
| RS-01 | P0 | [Declarar contratos de VAME y supervisados; aportar resolver científico](https://github.com/munhof/RAINSTORM/issues/2) | Implementado con regresiones; issue abierto para revisión |
| RS-02 | P0 | [Diagnosticar referencias degeneradas y versionar políticas de pose inválida](https://github.com/munhof/RAINSTORM/issues/3) | Pendiente |
| RS-03 | P0 | [Mostrar estado de reconstrucción y diferencias al cargar recetas](https://github.com/munhof/RAINSTORM/issues/4) | Pendiente |
| RS-04 | P1 | [Portar y verificar filtrado, interpolación y suavizado de Tesis_Facu](https://github.com/munhof/RAINSTORM/issues/5) | Pendiente |
| RS-05 | P1 | [Verificar unidades, ejes, ventanas, ROI y alineación por etapa](https://github.com/munhof/RAINSTORM/issues/6) | Pendiente |
| RS-06 | P1 | [Recuperar configuración final del VAME nativo y verificar equivalencia](https://github.com/munhof/RAINSTORM/issues/7) | Pendiente |
| RS-07 | P1 | [Aislar proyectos oficiales y comprobar fuentes, particiones y discretizador](https://github.com/munhof/RAINSTORM/issues/8) | Pendiente |
| RS-08 | P1 | [Preflight de recursos y diagnóstico escalonado de estabilidad GPU](https://github.com/munhof/RAINSTORM/issues/9) | Pendiente |
| RS-09 | P1 | [Inferencia del benchmark protegido y correspondencia de tareas](https://github.com/munhof/RAINSTORM/issues/10) | Pendiente |
| RS-10 | P2 | [Migrar controles y previews científicos al plugin RAINSTORM](https://github.com/munhof/RAINSTORM/issues/11) | Pendiente |
| RS-11 | P2 | [Configuración gráfica de variantes de modelos soportadas](https://github.com/munhof/RAINSTORM/issues/12) | Pendiente |

| ID | Prioridad | Issue definitivo | Estado de esta entrega |
|---|---|---|---|
| ST-01 | P0 | [Implementar contratos de entrada y validación compartida antes de encolar](https://github.com/munhof/STORM/issues/2) | Implementado con regresiones; issue abierto para revisión |
| ST-02 | P0 | [Separar plan activo de variantes de ejecución y evitar ramas heredadas](https://github.com/munhof/STORM/issues/3) | Pendiente |
| ST-03 | P0 | [Permitir dataset y preparación propios por rama](https://github.com/munhof/STORM/issues/4) | Pendiente |
| ST-04 | P1 | [Reutilizar preparación compatible y mostrar motivos de cache hit/miss](https://github.com/munhof/STORM/issues/5) | Pendiente |
| ST-05 | P1 | [Retener checkpoints históricos y validar continuidad](https://github.com/munhof/STORM/issues/6) | Pendiente |
| ST-06 | P1 | [Normalizar avance por fase, lote y sesión; separar heartbeat de progreso](https://github.com/munhof/STORM/issues/7) | Pendiente |
| ST-07 | P1 | [Validar tareas, métricas y compatibilidad antes de comparar](https://github.com/munhof/STORM/issues/8) | Pendiente |
| ST-08 | P2 | [Extender catálogo y hosts de UI mediante descriptores de plugins](https://github.com/munhof/STORM/issues/9) | Pendiente |
| ST-09 | P2 | [Editor gráfico con validación, teclado y equivalencia con especificaciones Python](https://github.com/munhof/STORM/issues/10) | Pendiente |
| ST-10 | P2 | [SDK de extensiones y publicación opcional de código en workers aislados](https://github.com/munhof/STORM/issues/11) | Pendiente |
| ST-11 | P3 | [Dashboards con gráficos, tablas y evidencia sincronizada](https://github.com/munhof/STORM/issues/12) | Pendiente |
| ST-12 | P3 | [Exportación, restauración y aceptación desde instalación limpia](https://github.com/munhof/STORM/issues/13) | Pendiente |


ST-01 y RS-01 se implementan juntos. ST-03 depende de ST-01/ST-02; RS-04 y RS-05
preceden RS-06 y RS-09. RS-07 depende de ST-03. ST-08 y RS-10 preceden ST-09;
ST-11 depende de alineación y comparación verificadas. GPU Hang se trata por
RS-08 independientemente del preprocesado. Dependencias detalladas y aceptación
están en cada issue.

La primera entrega implementa contratos y preflight; reconstrucción científica,
plan activo/variantes, inputs por rama, editor gráfico y dashboards permanecen
pendientes. No se modifican revisiones históricas ni se inician entrenamientos.

Contratos v1 y límites están en [reconstrucción](../reconstruction/README.md),
[worker/compatibilidad](../reconstruction/worker-adapters.md) y
[aceptación](../reconstruction/acceptance-status.md).
[Protocolo compartido](../../CONTRIBUTING.md).

## Propuesta original fechada (histórica, anterior a la entrega de contratos)

Los IDs P0/P1/P2/P3 siguientes son referencias de la propuesta; no sustituyen
los tickets ST/RS definitivos. La reconstrucción del estudio mediante nuevas
revisiones descrita abajo sigue pendiente y no se ejecuta en esta entrega.


Fecha: 2026-10-02. Estado: propuesta pendiente de implementación. [Diagnóstico y evidencia](01_diagnostico_estudios_20261002T220735.md).

## Objetivo de entrega

Un investigador registra datos, verifica preparación por etapa, configura ramas compatibles, entrena o importa modelos, recupera una ejecución e infiere sobre un benchmark protegido. La UI y Python producen la misma especificación. STORM mantiene los contratos generales; RAINSTORM aporta ciencia y herramientas de pose.

## Orden de trabajo

1. **Corregir el estudio y bloquear configuraciones inválidas:** validar antes de encolar, separar plan activo de variantes y diagnosticar pose inválida.
2. **Reconstruir la preparación:** portar etapas reales y probarlas contra Tesis_Facu; reutilizar resultados compatibles y revisar inputs por rama.
3. **Estabilizar entrenamiento e inferencia:** recetas completas, recursos, proyectos aislados, checkpoints y benchmark.
4. **Consolidar extensión gráfica:** contratos de componentes y editor de pipelines/variantes; código y UI comparten validación.
5. **Cerrar el recorrido:** dashboards, exportación y aceptación desde instalación limpia.

Cada ticket comienza con una prueba de regresión que falla por el problema identificado. Aplicar cambios locales, conservar tests existentes y abrir cambios de arquitectura solo cuando el contrato del ticket lo requiera. No lanzar entrenamientos largos como primera verificación.

## P0 — Recuperar un flujo correcto

| ID | Capa | Cambio concreto | Prueba y aceptación | Depende de |
|---|---|---|---|---|
| P0-01 | STORM engine + Studio | Validador de plan compartido que compruebe preparación permitida, pasos requeridos, configuración y entrada de cada rama antes de submit. Usar capacidades actuales para el primer fix. | Plan con oficial y pasos externos da error de campo/rama y crea cero jobs desde UI; Python aplica el mismo rechazo. Plan válido sí se ejecuta. | — |
| P0-02 | STORM Studio | Separar plan activo de variantes de ejecución. Primera corrección debe impedir que variantes reemplacen la raíz; incorporar referencia explícita/versionada y migración compatible. Quitar ramas heredadas de las variantes ejecutables. | Ejecutar 4 ramas mantiene el modelo y revisión elegidos al recargar. Reintentar una rama crea una sola corrida. Restaurar una receta la activa explícitamente. | — |
| P0-03 | RAINSTORM | Diagnóstico de confianza y referencias degeneradas por sesión/frame. Definir políticas versionadas; no reemplazar silenciosamente la receta histórica por orientación identity. | Fixture con inicio de baja confianza y huecos consecutivos reproduce el error actual; diagnóstico reporta sesión, frames y causa. Preview muestra afectados y exige política válida. | — |
| P0-04 | RAINSTORM + host de UI STORM | Exponer estado de reconstrucción y diferencias con tesis al cargar preset. Preservar parámetros de variantes y exigir confirmación explícita de cambios de receta, mostrando un diff. | Cargar oficial final produce steps=[] y KMeans/lr 0,0005; cambiar de preset no oculta cambio de dataset, backend o memoria. Historial recuperable. | P0-01, P0-02 |
| P0-05 | STORM Studio | Preparación y dataset propios por rama, compatibles con el plan antiguo. Evitar duplicar datos: referenciar revisiones existentes. Ocultar componentes de demostración salvo catálogo avanzado. | Native usa preparado/ventanas; oficial usa H5 completo sin pasos externos; ambos conservan la cohorte y particiones originales. Identity/constant no aparecen como modelos científicos recomendados. | P0-01, P0-02 |

**Salida P0:** ninguna de las cuatro configuraciones reportadas llega al worker sin diagnóstico previo. No implica que ya se haya reconstruido científicamente VAME ni solucionado ROCm.

## P1 — Equivalencia científica, recuperación y ejecución

| ID | Capa | Cambio concreto | Prueba y aceptación | Depende de |
|---|---|---|---|---|
| P1-01 | RAINSTORM | Portar NaN por confianza, filtrado anatómico/velocidad, interpolación PCHIP/relleno de bordes, mediana 5 y Gaussian smoothing. Registrar unidades, ejes, parámetros y orden reales. | Comparar salidas intermedias con funciones de Tesis_Facu sobre fixtures y sesiones representativas; tolerancias justificadas, conteos y máscaras iguales. Nada atraviesa sesiones, discontinuidades o límites protegidos. | P0-03 |
| P1-02 | RAINSTORM | Recuperar el contrato temporal completo: shape, orden de partes, frame central, offsets, recorte, futuro opcional y ROI/contexto. Recetas distintas para nativo y oficial. | Comparar cada etapa con builder y notebooks finales; conservar correspondencia al frame original y justificar 20 frames nativos frente al time_window19 oficial sin asumir equivalencia. | P1-01 |
| P1-03 | STORM + RAINSTORM | Identidad semántica y reutilización de preparación: fuentes/hashes, particiones, reservas, configuración resuelta, versiones, unidades y transformaciones ajustadas. Explicar cache hit/miss. Aplicar primero al camino inline que vuelve a procesar. | Dos recetas equivalentes reutilizan artefacto íntegro; cambios de reservas/partición o versión invalidan. Revisión 12 solo se ofrece cuando sea compatible y se identifica su receta parcial. | P0-05, P1-01 |
| P1-04 | STORM host + RAINSTORM renderer | Recuperar y mostrar preview de cada etapa y dataset final; seleccionar misma sesión/frame antes/después, tabla descargable, máscara y estadísticas de calidad. | Navegador muestra datos originales, filtrados y orientados alineados; ventanas se visualizan usando frame central, sin inventar coordenadas. Recuperación no exige reprocesar. | P1-02, P1-03 |
| P1-05 | RAINSTORM | Auditar y portar configuración final de red/trainer nativo; decidir explícitamente qué opciones históricas son necesarias. Versionar arquitectura, pérdida, normalización si existía, clipping, KL y discretizador. | Equivalencia de forward/loss con pesos controlados para arquitectura correspondiente; entrenamiento corto repetible. Cada diferencia histórica pendiente aparece en el manifest. | P1-02 |
| P1-06 | RAINSTORM | Aislar proyecto oficial por ejecución/bundle y verificar su identidad antes de reutilizar config/modelos/discretizador. Enlazar fuentes completas y particiones externas; distinguir validación interna de evaluación reservada. | Dos planes/datasets no comparten accidentalmente proyecto. Fixture reservado/cortado/discontinuo se rechaza con motivo o usa segmentación explícita probada. Inferencia no reajusta discretizador. | P0-05, P1-02 |
| P1-07 | STORM recursos + RAINSTORM runtime | Preflight de RAM/VRAM y backend; materialización acotada, mapped disponible, prueba biológica GPU pequeña y luego escalada controlada. Disponibilidad GPU y estabilidad comprobada se muestran separadas. | Evitar asignación íntegra de ventanas que exceda presupuesto; registrar dispositivo/backend y prueba usada. GPU Hang cierra estado/ETA y conserva contexto. Pasar una prueba pequeña no habilita afirmaciones de estabilidad total. | P1-05 |
| P1-08 | STORM | Retener varios checkpoints válidos con manifest e identidad; diferenciar continuación completa de reinicialización desde pesos. Compatible con artefactos actuales y comprobación de receta/entorno. | Interrupción conserva última época y anterior; restaurar optimizador/RNG/progreso; no duplicar salidas. Resolver regresión abierta de retención en `test_learning_lifecycle.py`. | P1-05, P1-06 |
| P1-09 | STORM observabilidad + adapters RAINSTORM | Eventos estructurados por fase/subfase/sesión/lote, correlación ejecución/componente, error y último punto seguro. Reflection instrumenta límites; callbacks reportan avance interno. | No quedan lotes/porcentajes de una etapa anterior; heartbeat no incrementa progreso; ETA solo con avance medido. Fallo y reanudación enlazados. | P0-01 |
| P1-10 | RAINSTORM + STORM comparación | Inferencia exclusiva del benchmark; task mapping versionado y métricas elegibles por salida. Clasificación binaria y motivos tienen evaluaciones distintas. | Crop1–7200, 5303 etiquetas válidas y frame mapping comprobados. Benchmark nunca participa en fit/calibración. No ranking binario vs12 clases sin correspondencia válida; ARI/NMI invariantes a permutación. | P1-05, P1-06 |

**Salida P1:** preparación verificada etapa por etapa; entrenamientos cortos y recuperación reales; inferencia alineada y comparación justificada. Entrenamiento final completo es una prueba de aceptación separada.

## P2 — Extensión gráfica y en código

| ID | Capa | Cambio concreto | Prueba y aceptación | Depende de |
|---|---|---|---|---|
| P2-01 | STORM engine | Ampliar el catálogo actual con esquemas de conectores/procesadores, puertos de entrada/salida y capacidades serializables. Adaptar clases existentes sin importar Django ni librerías científicas pesadas. | Plugin numérico genérico registra carga, proceso, modelo y preview; UI/Python/worker validan el mismo manifest. Registros anteriores siguen legibles. | P0-01, P0-05 |
| P2-02 | RAINSTORM + STORM hosts | Mover resolver `pose.*`, etiquetas, selectores corporales y previews a extensiones RAINSTORM. Studio consulta contratos/renderers genéricos. | STORM sin plugin funciona con datos numéricos; con RAINSTORM aparecen pose/ROI. Estudios viejos se leen y ejecutan con adapter de compatibilidad, sin duplicar lógica científica. | P2-01 |
| P2-03 | STORM Studio | Canvas por etapas y ramas con puertos, errores locales, undo, duplicar nodo/rama, guardar receta, restaurar y exportar spec. Formularios derivados de esquemas. | Pipeline simple se arma sin JSON; round-trip gráfico→spec→Python→gráfico conserva parámetros y hash; teclado y viewport reducido funcionales. | P2-01, P2-02 |
| P2-04 | RAINSTORM | Editor visual de variantes de modelos soportadas. VAME nativo: representación/discretizador y arquitectura admitida; oficial: proyecto y opciones reales. Importación de modelos con contrato de features. | Validar shape y salida antes de entrenar; variante visual produce configuración equivalente al builder Python. Funciones no implementadas aparecen pendientes o no se ofrecen. | P1-05, P1-06, P2-03 |
| P2-05 | STORM SDK + RAINSTORM ejemplos | Extensión por código mediante plantilla de paquete: clase/factory, schema, puertos, capabilities, tests, entorno y renderer opcional. Registro versionado y recarga controlada del worker. | Componente nuevo aparece en catálogo y funciona desde UI y Python; manifest enlaza hash/código/entorno. Ejemplo de procesador y modelo externos sin editar Studio. | P2-01 |
| P2-06 | STORM workers | Opcional: editor de código y publicación desde UI, ejecutando pruebas y preview en runtime aislado antes de registrar una versión. | Fallo del plugin no afecta Django ni sobrescribe versiones/artefactos; permisos de runtime visibles y ejecución cancelable. Un paquete válido queda disponible como nodo. | P2-05 |

El editor de código dentro de la UI es posterior al registro de paquetes y al editor gráfico. La primera versión gráfica compone implementaciones reales; no genera adapters ficticios.

## P3 — Dashboards y distribución

| ID | Capa | Cambio concreto | Prueba y aceptación | Depende de |
|---|---|---|---|---|
| P3-01 | STORM visualizaciones + RAINSTORM vistas científicas | Dashboard centrado en preguntas: qué datos fueron cubiertos, qué estados dominan, duración/transiciones, dónde discrepan modelos y qué excluyó la evaluación. Histogramas, matrices con colores y tablas crudas. | Gráficos y CSV coinciden; filtros sesión/segmento/partición/máscara afectan ambos. No mezclar IDs de discretizadores locales. Video y pose muestran mismo frame/estado. | P1-04, P1-10 |
| P3-02 | STORM + RAINSTORM | Exportar evidencia con manifests, configuraciones, fuentes/hashes, anotaciones, modelos y reportes; instalación desde clones limpios con versión fija. | Recorrido completo en workspace nuevo sin checkout de Tesis_Facu; revisión exportada conserva métricas y datos originales tras corregir el benchmark. | P1-08, P3-01 |

## Primera entrega revisable

Implementar P0-01, P0-02 y P0-03 en cambios separados y pequeños. Después corregir la configuración del estudio mediante **una nueva revisión**, con diff visible respecto de 71 y ramas sin compartir preparación incompatible. Mantener todos los fallos históricos.

Para VAME nativo, ofrecer dos rutas claramente identificadas: artefacto parcial existente compatible, para diagnóstico operativo; receta reconstruida y verificada de P1, para resultados científicos. Para oficial, usar H5 de sesiones completas, preparación interna y receta final explícita. No iniciar “todas las ramas” hasta que la validación pase por rama.

## Migración y preservación

- La revisión 71 y las corridas 72–75 permanecen inmutables; ninguna se convierte en éxito por editar su payload.
- El plan activo se elige explícitamente. En estudios antiguos sin ese dato, inferir candidatos de raíz y mostrar ambigüedad para resolverla; no tomar silenciosamente la última variante.
- Conservar recetas legacy, IDs, artefactos y correcciones. Migrar mediante versiones/adapters de lectura.
- Cambiar filtrado, política degenerada o arquitectura crea versión científica nueva e invalida únicamente resultados cuya identidad cambió.
- La reutilización registra artefacto origen, compatibilidad y motivo; una recuperación fallida no implica automáticamente recalcular todo.

## Criterio de cierre

1. Crear el estudio desde UI y cargar los 86 H5/ROI con particiones históricas comprobadas.
2. Recuperar una receta y ver salida de cada etapa; procesar y reutilizar sin aplicar dos veces transformaciones.
3. Mantener plan activo al ejecutar ramas nativa/oficial con sus entradas correctas.
4. Entrenar con configuración y entorno registrados; interrumpir y recuperar desde checkpoint válido.
5. Importar los dos supervisados, ejecutar los modelos terminados sobre benchmark protegido y mostrar cobertura/métricas compatibles.
6. Corregir anotación sin modificar evidencia previa; comparar y exportar antecedentes.
7. Extender un pipeline gráficamente y registrar un componente Python sin cambiar código específico de STORM Studio.

No declarar cierre por navegación HTTP200, test sintético GPU o inferencia de modelos de diagnóstico. Se requiere evidencia del recorrido científico completo.
