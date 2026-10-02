# Procedencia de los supervisados

## Evidencia recuperada

`notebooks/original/3a-Create_models.ipynb` describe entrenamiento para reconocer
exploración usando posiciones del animal y un objeto objetivo. Sus celdas 10–11
preparan anotaciones, seleccionan frames cerca de exploración y crean una
partición de entrenamiento, test y validación. Las celdas 15 y 25 describen,
respectivamente, una entrada de un frame y una secuencia temporal.

En el commit histórico `0523b9a35c070cbe9d6e8f45858d4cae94cc36db`,
`backend/modeling/data_handling.py::prepare_data` selecciona coordenadas de
partes corporales y del objetivo. Construye el target promediando las columnas
de los anotadores configurados, aplica suavizado mediano y gaussiano y luego
`apply_sigmoid_transformation`: `1 / (1 + exp(-9 * (value - 0.6)))`, redondeado
a tres decimales, con valores hasta 0.3 llevados a 0 y desde 0.9 llevados a 1.
Son targets continuos transformados; una salida sigmoide no demuestra por sí
misma calibración probabilística.

`examples/models/colabels.csv` contiene cinco columnas de anotadores,
coordenadas `tgt_x/tgt_y` y posiciones corporales. El notebook lo presenta como
un ejemplo de reconocimiento de objetos. La presencia de ese archivo no prueba
que los dos bundles distribuidos fueran entrenados exactamente con él.

## Lo que sigue sin verificarse

Los archivos `example_simple.keras` y `example_wide.keras` tienen firmas
verificadas de 12 features y ventanas de 7 × 12, respectivamente, con una salida
binaria. No se recuperó un manifiesto que enlace sus hashes con los parámetros,
las listas de sesiones y la partición usados para entrenar esos pesos.
Por eso `training_population` sigue siendo desconocida y no se cambia
automáticamente la semántica declarada de los bundles.

Explorar un objeto es una tarea distinta de identificar si ese objeto es
conocido o nuevo. El benchmark NOR distingue `Known/Novel`; el CSV humano de
Manual Labels tiene 12 categorías. Para calcular métricas de clasificación,
el investigador debe declarar una correspondencia versionada con la tarea real
del modelo y justificarla. No corresponde asignar `1 = Novel` por el nombre
del experimento ni presentar estos modelos en un ranking de 12 conductas.

La receta histórica recuperada aporta evidencia de intención y procesamiento;
la procedencia de cada artefacto entrenado requiere evidencia adicional.
