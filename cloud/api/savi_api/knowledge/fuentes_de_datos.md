# Fuentes de datos

SAVI v3 usa exclusivamente datos reales de Ames, Iowa, integrados por número de parcela (PID).

## El problema del CSV combinado (v2)

Una auditoría previa encontró que el archivo `ames_combined_2006_2024.csv` usado en la versión
anterior (v2) del sistema **no era utilizable**: el 93% de sus filas ("ventas 2024") en realidad
venían del padrón del assessor (que tiene 25 columnas reales) rellenado con 61 variables
constantes inventadas (por ejemplo `Neighborhood=NAmes` o `LotArea=9000` para todas las filas), y
su columna `SalePrice` era en realidad el **avalúo fiscal 2024**, no el precio de una venta real.
El R²=0.96 que reportaba v2 se apoyaba en ese artefacto defectuoso y no era un resultado válido.

## Fuentes reales usadas en v3

| Fuente | Cobertura | Uso |
|---|---|---|
| De Cock (2011), ventas residenciales de Ames | 2,930 ventas 2006-2010, con PID | Target real del AVM (precio de venta) |
| Ames City Assessor — padrón residencial 2024 | 18,931 parcelas (25 columnas reales) | Características actuales de cada parcela + cartera que evalúa el agente |
| FHFA All-Transactions HPI, Ames MSA (CBSA 11180) | Trimestral, hasta 2026T2 | Llevar todos los precios y avalúos a dólares de "hoy" |
| Zillow Home Value Index (ZHVI), Ames, mid-tier SA | Mensual | Validación independiente del nivel de mercado (no se usa para entrenar) |
| Ames City Assessor — ventas 2020-2022 (opcional) | Descarga manual, esquema variable | Ventas recientes adicionales si están disponibles en `reference/` |
| Coordenadas (tidymodels `modeldata::ames`) | Lat/lon por PID de venta De Cock | Geolocalización de ventas y, por agregación (subdivisión → área), de las parcelas del padrón que no se vendieron |

De las 2,930 ventas De Cock, 2,891 tienen correspondencia en el padrón 2024. Tras filtrar
condiciones de venta no comerciales (se mantienen "Normal" y "Partial") y descartar las casas
cuya área habitable cambió más de 20% desde la venta (el padrón refleja el estado *actual* de la
parcela, no el que tenía al momento de venderse), quedan **2,512 ventas** de entrenamiento real.

## Integración y ajuste temporal

Todo se une por PID (número de parcela, formato de 10 dígitos). Los precios de venta y los
avalúos del padrón 2024 se llevan a dólares del trimestre más reciente del HPI disponible
(`hpi_latest`, p.ej. 2026T2) multiplicando por la razón `HPI(hoy) / HPI(trimestre original)`.
Esto permite comparar en la misma base monetaria una venta de 2006 con el avalúo de 2024.

## Geolocalización

No existen direcciones en los datos del assessor. Las coordenadas de una parcela se asignan así,
en orden de precisión decreciente (campo `geo_precision`): `exact` si la parcela se vendió en
De Cock y tiene coordenada propia; `subdivision` (centroide de las ventas geolocalizadas de su
misma subdivisión) si no; `map_area` (centroide a nivel de township/sección) si tampoco hay
subdivisión con datos; `none` si no hay ninguna referencia geográfica disponible para esa zona.
