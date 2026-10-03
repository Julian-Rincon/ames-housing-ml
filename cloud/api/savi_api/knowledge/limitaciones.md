# Limitaciones conocidas del sistema

## Las características reflejan el estado 2024, no el de la venta

El padrón del Ames City Assessor describe cómo es la parcela **hoy** (2024): si una casa se
remodeló, se le agregó un garaje o cambió de condición después de una venta De Cock
(2006-2010), el modelo ve las características actuales, no las que tenía al momento de esa
venta. Para mitigar el peor caso, el pipeline descarta de entrenamiento las ventas cuya área
habitable actual difiere más de 20% de la que tenía al venderse — pero el resto de atributos
(grado, condición, baños, garaje, etc.) no se verifica contra el momento de la venta.

## Ajuste de precios por HPI, no por el mercado específico de cada parcela

Todos los precios y avalúos se llevan a dólares de "hoy" multiplicando por la razón entre el HPI
(índice de precios de vivienda) del trimestre actual y el del trimestre original. Este es un
ajuste de **nivel de mercado agregado** para el área metropolitana de Ames; no captura
variaciones de precio específicas de un barrio, un segmento de vivienda, o eventos locales entre
el momento de la venta original y hoy.

## Sin direcciones reales

No hay direcciones postales en ningún archivo fuente. La geolocalización (`lat`/`lon`) es exacta
sólo para las parcelas que efectivamente se vendieron en el periodo De Cock con coordenadas
conocidas (campo `geo_precision="exact"`); para el resto se aproxima con el centroide de su
subdivisión o de su área (township/sección), lo que introduce imprecisión geográfica,
especialmente en zonas grandes o heterogéneas.

## GPU denegada en el entorno de entrenamiento

El entorno de AWS Academy Learner Lab usado para entrenar deniega explícitamente instancias
`ml.g4dn.*` (GPU) en SageMaker por política IAM, aunque la cuota de servicio la permita. El
entrenamiento del Double DQN cae automáticamente a CPU (`ml.m5.xlarge`); el código es el mismo
(usa `torch.cuda.is_available()`), pero en una cuenta sin esa restricción el entrenamiento sería
más rápido en GPU. Esto no afecta la calidad del modelo final, sólo el tiempo de entrenamiento.

## Conjunto de prueba pequeño → intervalos de confianza amplios

El conjunto TEST es sólo el 15% de las 2,512 ventas de entrenamiento (del orden de cientos de
observaciones). La recompensa media reportada sobre TEST viene siempre acompañada de un
intervalo de confianza bootstrap al 95%; estos intervalos suelen ser relativamente anchos, y las
diferencias entre reglas de decisión (p.ej. contra la política base de Value Iteration) a veces
no son estadísticamente significativas, aun cuando la media observada favorezca a una regla.
Cualquier cifra de recompensa o comparación entre políticas debe citarse junto con su intervalo
de confianza, no sólo el promedio puntual.

## El AVM no es perfecto

El MAPE (error porcentual absoluto medio) del AVM out-of-fold está típicamente en el orden de
7-8%, y su MAE (error absoluto medio) en decenas de miles de dólares. Las valuaciones "what-if"
de la API usan ese mismo modelo y heredan su incertidumbre; por eso `value_property` siempre
reporta un rango (± MAPE) además del valor puntual, y el agente de RL existe precisamente para
decidir cuándo esa incertidumbre amerita revisión humana o rechazo en vez de aprobación directa.
