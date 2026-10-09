# Estudio 8: aceptación de contextos v2

La evidencia actual es `study8-context-acceptance.json`: revisión científica 85,
run 86 y job `332a35b1-637a-4f01-804b-fcfd6d46241e` completado en Studio.
La revisión de datos 11 y las corridas históricas permanecen intactas.

Se prepararon todas las 62 sesiones train y 2 validation. La calibración explícita
usa ROI histórico, escala 15,02 px/cm y altura 1052 con referencia/hash en el grafo.
Las 22 sesiones test permanecieron reservadas; sus archivos no se abrieron.

`study8-stage-comparison.json` compara confianza, velocidad, máscara espacial,
interpolación y suavizado con las funciones recuperadas de Tesis_Facu en las 64
sesiones. Todas las máscaras coinciden y todas las coordenadas están dentro de
atol=1e-6, rtol=1e-6 bajo esta misma calibración. Esto no demuestra que la corrida
histórica nativa usara ese ROI ni esa calibración.

VAME nativo entrenó una época CPU con 256 ventanas deterministas, semilla 156,
50 estados, latent 20, hidden 128, batch 256, LR 0,001 y KL 0,5. Produjo 37.743
predicciones sobre las dos sesiones de validación. Pérdida total 7,076162815;
reconstrucción 6,390771389; KL 1,370783091. Es evidencia de ejecución, no de calidad.
Los resultados y el modelo se conservan en el artifact store local con digest;
las rutas absolutas del JSON describen este workspace y no son una exportación portable.

Reproducir desde el entorno científico CPU con `scripts/prepare_study8_context.py`,
`scripts/compare_study8_reconstruction.py` y `scripts/accept_study8_context.py`.
Los parámetros `--help` describen rutas requeridas. Cada corrida nueva recibe
identidad propia y no sobrescribe los artefactos de corridas anteriores.

`study8-context-audit.json` conserva el diagnóstico anterior a preparar/entrenar;
su estado de bloqueo es histórico y queda superado por esta evidencia.
Continúan sin certificar equivalencia completa de estudio/modelo histórico,
warmup KL histórico, estabilidad GPU y entrenamiento completo. La receta v2
rechaza segmentos totalmente inválidos y no fabrica coordenadas cero ni identidad
en una orientación degenerada.

El contrato temporal del plugin declara reloj pose_frame, unidad frame, origen
source_frame_id y alcance por sesión. El muestreo conserva timestamps e IDs.
Las fuentes en segundos requieren una transformación explícita antes de unir.

## Comparación exploratoria en Studio · 2026-10-08

La revisión 91 del estudio 8 guarda un grafo de 15 nodos. La corrida
`a77c6f5d-31ea-43e4-8608-9f5238e0233a` completó sus cinco salidas sobre las
2 sesiones de validación del manifiesto original de 86 sesiones (62 train,
2 validation y 22 test). El test reservado no se abrió. El dataset NOR activo
en el estudio no reemplaza las fuentes fijadas en este grafo.

| Salida | Observaciones | Resultado |
| --- | ---: | --- |
| VAME recuperado | 37.743 | 45 estados; predicciones idénticas a la corrida guardada |
| Constant, valor 0 | 37.743 | Una salida fija, como control |
| Identity, control de VAME | 37.743 | Reproduce las etiquetas de VAME; no es un modelo independiente |
| Supervisado simple | 34.635 | Clase 0 en todos los frames admitidos |
| Supervisado temporal | 31.512 | Clase 0 en todas las ventanas admitidas |

VAME reutiliza pesos y contextos preparados de la corrida
`332a35b1-637a-4f01-804b-fcfd6d46241e`, sin reentrenamiento. Los supervisados
usan los bundles existentes, con orden explícito nose, left_ear, right_ear,
head, neck, body. La receta exploratoria conserva píxeles, descarta frames con
coordenadas no finitas o confianza inferior a 0,6 y no interpola ni rellena.
Simple consume un frame; wide consume offsets -3 a 3, sin atravesar descartes
ni sesiones. No se afirma que esta receta coincida con el entrenamiento histórico.
Sus clases siguen con significado y población de entrenamiento desconocidos.

Hay 31.498 observaciones comunes por sesión e identidad. La comparación muestra
frecuencias sobre la cohorte propia de cada salida; no recalcula métricas ni
habilita ranking entre estados de clustering y clases binarias sin equivalencia.
Las salidas degeneradas de los supervisados se conservan como evidencia, no como
validación de su calidad.

El grafo exportable y la aceptación están en
`.storm/rainstorm/study8-comparison/experiment.json` y `acceptance.json` dentro
del workspace STORM. Studio ofrece las distribuciones conjuntas en
`/studies/8/comparisons/?jobs=a77c6f5d-31ea-43e4-8608-9f5238e0233a`.
También se corrigió la inferencia VAME recuperada para respetar `num_threads`
del checkpoint; la corrección tiene una regresión con PyTorch.
