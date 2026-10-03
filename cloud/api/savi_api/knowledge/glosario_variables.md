# Glosario de variables

## Sistema de grado de construcción (manual del assessor de Iowa)

El campo `grade` (p.ej. `"4+5"`, `"2-10"`) combina dos partes:

- **Grado base** (`grade_base`, 1 a 6): calidad y costo de construcción de la estructura, de
  menor (1, mínima) a mayor (6, muy superior/personalizada).
- **Modificador** (`grade_mod`): un ajuste porcentual sobre el grado base, expresado como un
  entero con signo que normalmente va de −10 a +10 en pasos de 5 (es decir, ±5% o ±10% sobre el
  costo base de ese grado). Por ejemplo `"4+5"` es grado base 4 con un +5% de ajuste; `"2-10"`
  es grado base 2 con un −10% de ajuste (construcción algo por debajo del estándar de ese grado).

## Escala de condición

El campo `condition_label` (texto) se mapea a un valor numérico 0-8 (`condition`) en esta escala
ordinal: `Very Poor=0, Poor=1, Fair=2, Below Normal=3, Normal=4, Above Normal=5, Good=6,
Very Good=7, Excellent=8`. Valores más altos indican mejor estado de conservación.

## `map_area`: proxy geográfico público

`map_area` son los 4 primeros dígitos del PID (número de parcela) del assessor de Ames, que
codifican el township y la sección catastral. Se usa como proxy geográfico público (no hay
direcciones en los datos) y entra al AVM como una variable categórica one-hot (`area_<código>`,
36 áreas distintas en la cartera). Es un nivel de agregación más fino que el barrio (`neighborhood`,
28 barrios De Cock) pero sin nombre descriptivo propio.

## Tipo y condición de sótano

`basement_type` (texto del assessor) se traduce a una fracción `bsmt_frac` del área del sótano
respecto al área típica: `Full=1.0, 3/4=0.75, 1/2=0.5, 1/4=0.25` (y una categoría especial
`"Bsmt SF (Obsv)"=0.5`); cualquier otro valor (incluido "sin sótano") se trata como `0.0`.
`bsmt_finished` es 1 si el sótano está terminado (`BASEMENT FINISH = "Yes"`), 0 si no.

## Garaje

`garage_type` (texto) determina tres variables: `has_garage` (1 si hay algún tipo de garaje
registrado), `garage_attached` (1 si el texto empieza con "Att" — adosado — o es un carport
adosado), y `garage_carport` (1 si el texto contiene "Carport"). `n_garages` cuenta cuántos
garajes distintos tiene registrados la parcela (una misma parcela puede tener más de una fila en
el padrón original cuando tiene varios garajes).

## Estilo de la vivienda

El campo `style` (p.ej. "1 Story Frame", "2 Story Brick") se traduce a: `stories` (número de
pisos, admite medios pisos como "1 1/2 Story" → 1.5), y cuatro indicadores binarios derivados del
texto: `is_split` (diseño "split level"), `is_condo` (condominio), `is_townhouse` (casa adosada en
hilera) e `is_brick` (construcción en ladrillo).

## Otras variables numéricas

`living_area` (pies cuadrados de área habitable total), `rooms_above`/`rooms_below` (habitaciones
sobre/bajo el nivel del suelo), `beds_above`/`beds_below` (dormitorios sobre/bajo el nivel del
suelo), `n_plumbing` (artefactos de plomería), `n_fireplaces` (chimeneas), `n_additions`
(adiciones a la estructura), `n_porches`/`n_decks` (porches/terrazas), `attic_finished` (ático
terminado), `land_value_2024`/`assessed_2024` (avalúo fiscal 2024 del terreno/total, en dólares
nominales de ese año) y `log_land_value` (logaritmo natural de `1 + valor del terreno en dólares
de hoy`, la transformación que efectivamente usa el AVM como variable de entrada).
