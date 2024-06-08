# # Practica Final
# Modulo 4 - Máster Data Science y Business Analytics
# Sergio Hervás Aragón

# ### Librerías

import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import warnings
warnings.filterwarnings('ignore')

# ## 1. EDA

# ### 1.1. Carga de datos

file = './caso_final_small_20k_con_mes.csv'
df = pd.read_csv(file, sep=',')

df.head()

print(u'- El número de filas en el dataset es: {}'.format(df.shape[0]))
print(u'- El número de columnas en el dataset es: {}'.format(df.shape[1]))
print(u'- La variable objetivo es: {}'.format(df.columns[-1]))
print(u'- Los nombres de las variables independientes son: {}'.format(list(df.columns[:-1])))

# ### 1.2. Descripción y Análisis

df.info()

# ### 1.3. Variables continuas

df.describe()

valores_distintos = df.select_dtypes(include=['float64', 'int']).nunique()
valores_distintos

# Crear scatter plots con líneas de tendencia
fig, axes = plt.subplots(nrows=7, ncols=5, figsize=(20, 35))
axes = axes.flatten()
columnas_numeric = df.select_dtypes(include=['float64', 'int']).columns
columnas_numeric = columnas_numeric.drop(['MES', 'TARGET'])
colors = [color['color'] for color in plt.rcParams['axes.prop_cycle']]

for i, col in enumerate(columnas_numeric):
    sns.regplot(
        data=df,
        x=col,
        y='TARGET',
        color=colors[i % len(colors)],
        line_kws={'color': 'lightblue', 'linewidth': 2},
        scatter_kws={'alpha': 0.3}
    , ax=axes[i])
    axes[i].set_title(col, fontsize=7, fontweight="bold")
    axes[i].tick_params(labelsize=6)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    axes[i].grid(False)
plt.show()

# Tipos de gráficas a observar:
# 1. Positivas: Gráficas donde a medida que aumenta la variable independiente ('TARGET'), el valor de la variable aumenta
#     - MB_TOTALES (Cuando más MB totales consumidos, más posivilidad de adquirir un producto adicional)
#     - FACTURACION_TOTAL_IMPUESTOS (Cuando más facturación total de impuestos, más posivilidad de adquirir un producto adicional)
#     - FACTURACION_CUOTA (Cuando más facturación haya, más posivilidad de adquirir un producto adicional)
#     - [...]
# 
# 2. Negativas: Gráficas donde a medida que disminuye la variable independiente, el valor de la variable aumenta, lo que provoca un cliente no adquiera un producto adicional:
#     - EDAD (Cuando más edad, menos posivilidad de adquirir un producto adicional)
#     - NUM_DIAS_ACTIVO (Cuantos mas dias activos, menos posivilidad de adquirir un producto adicional)
#     - DIA_PRIMERA_CUENTA (Cuatos mas dias pasen con una cuenta recien echa,  menos posivilidad de adquirir un producto adicional)

# ### 1.4. Tratamiento de nulos

msno.matrix(df)

imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
msno.matrix(df)

# ### 1.5. Distribucion de la variable objetivo

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='TARGET')
plt.title('Distribución de Supervivencia')
plt.xlabel('Supervivencia')
plt.ylabel('Frecuencia')
plt.grid(False)
plt.show()

# ### 1.6. Outliers

fig, axes = plt.subplots(nrows=7, ncols=5, figsize=(25, 35))
axes = axes.flatten()
colors = [color['color'] for color in plt.rcParams['axes.prop_cycle']]

for i, col in enumerate(columnas_numeric):
    sns.boxplot(
        data=df,
        y=col,
        x='TARGET',
        color=colors[i % len(colors)],
        ax=axes[i]
    )
    axes[i].set_title(col, fontsize=7, fontweight="bold")
    axes[i].tick_params(labelsize=6)
    axes[i].set_ylabel(col, fontsize=8)
    axes[i].set_xlabel('TARGET', fontsize=8)
    axes[i].grid(False)
plt.show()

df_outliers = df.select_dtypes(include=['float64', 'int64'])
outlier_indices_above = []
outlier_indices_below = []
for column in df_outliers.columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_above = df[column] > upper_bound
    outliers_below = df[column] < lower_bound
    
    outlier_indices_above.extend(outliers_above[outliers_above].index)
    outlier_indices_below.extend(outliers_below[outliers_below].index)

outlier_indices_above = pd.DataFrame(outlier_indices_above, columns=['Index'])
outlier_indices_below = pd.DataFrame(outlier_indices_below, columns=['Index'])

outlier_indices = pd.concat([outlier_indices_above, outlier_indices_below], axis=0)

df = df.drop(index=outlier_indices['Index']).reset_index(drop=True)

print(len(df))


fig, axes = plt.subplots(nrows=7, ncols=5, figsize=(25, 35))
axes = axes.flatten()
colors = [color['color'] for color in plt.rcParams['axes.prop_cycle']]

for i, col in enumerate(columnas_numeric):
    sns.boxplot(
        data=df,
        y=col,
        x='TARGET',
        color=colors[i % len(colors)],
        ax=axes[i]
    )
    axes[i].set_title(col, fontsize=7, fontweight="bold")
    axes[i].tick_params(labelsize=6)
    axes[i].set_ylabel(col, fontsize=8)
    axes[i].set_xlabel('TARGET', fontsize=8)
    axes[i].grid(False)
plt.show()

df_outliers = df.select_dtypes(include=['float64', 'int64'])
outlier_indices_above = []
outlier_indices_below = []
for column in df_outliers.columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_above = df[column] > upper_bound
    outliers_below = df[column] < lower_bound
    
    outlier_indices_above.extend(outliers_above[outliers_above].index)
    outlier_indices_below.extend(outliers_below[outliers_below].index)

outlier_indices_above = pd.DataFrame(outlier_indices_above, columns=['Index'])
outlier_indices_below = pd.DataFrame(outlier_indices_below, columns=['Index'])

outlier_indices = pd.concat([outlier_indices_above, outlier_indices_below], axis=0)

df = df.drop(index=outlier_indices['Index']).reset_index(drop=True)

print(len(df))


fig, axes = plt.subplots(nrows=7, ncols=5, figsize=(25, 35))
axes = axes.flatten()
colors = [color['color'] for color in plt.rcParams['axes.prop_cycle']]

for i, col in enumerate(columnas_numeric):
    sns.boxplot(
        data=df,
        y=col,
        x='TARGET',
        color=colors[i % len(colors)],
        ax=axes[i]
    )
    axes[i].set_title(col, fontsize=7, fontweight="bold")
    axes[i].tick_params(labelsize=6)
    axes[i].set_ylabel(col, fontsize=8)
    axes[i].set_xlabel('TARGET', fontsize=8)
    axes[i].grid(False)
plt.show()

df_outliers = df.select_dtypes(include=['float64', 'int64'])
outlier_indices_above = []
outlier_indices_below = []
for column in df_outliers.columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_above = df[column] > upper_bound
    outliers_below = df[column] < lower_bound
    
    outlier_indices_above.extend(outliers_above[outliers_above].index)
    outlier_indices_below.extend(outliers_below[outliers_below].index)

outlier_indices_above = pd.DataFrame(outlier_indices_above, columns=['Index'])
outlier_indices_below = pd.DataFrame(outlier_indices_below, columns=['Index'])

outlier_indices = pd.concat([outlier_indices_above, outlier_indices_below], axis=0)

df = df.drop(index=outlier_indices['Index']).reset_index(drop=True)
print(len(df))


fig, axes = plt.subplots(nrows=7, ncols=5, figsize=(25, 35))
axes = axes.flatten()
colors = [color['color'] for color in plt.rcParams['axes.prop_cycle']]

for i, col in enumerate(columnas_numeric):
    sns.boxplot(
        data=df,
        y=col,
        x='TARGET',
        color=colors[i % len(colors)],
        ax=axes[i]
    )
    axes[i].set_title(col, fontsize=7, fontweight="bold")
    axes[i].tick_params(labelsize=6)
    axes[i].set_ylabel(col, fontsize=8)
    axes[i].set_xlabel('TARGET', fontsize=8)
    axes[i].grid(False)
plt.show()


# df_outliers = df.select_dtypes(include=['float64', 'int64'])

# Q1 = df_outliers.quantile(0.25)
# Q3 = df_outliers.quantile(0.75)

# IQR = Q3 - Q1

# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR

# outliers_above = (df_outliers > upper_bound)
# outliers_below = (df_outliers < lower_bound)

# total_muestras = len(df_outliers)

# outliers = outliers_above.sum().sum() + outliers_below.sum().sum()

# porcentaje_outliers = (outliers / total_muestras) * 100

# print(f"Número total de muestras:{total_muestras}")
# print(f"Número de outliers totales:\n{outliers}")
# print(f"Porcentaje de outliers totales:\n{porcentaje_outliers}")

# outlier_indices_above = pd.DataFrame(outliers_above[outliers_above.sum(axis=1) > 0].index)
# outlier_indices_below = pd.DataFrame(outliers_below[outliers_below.sum(axis=1) > 0].index)

# outlier_indices = pd.concat([outlier_indices_above, outlier_indices_below], axis=1, ignore_index=True)

# df = df.drop(index=outlier_indices).reset_index(drop=True)

# print(len(df))

# # Crear el modelo LOF
# lof = LocalOutlierFactor(n_neighbors=20)
# df_outliers = df.select_dtypes(include=['float64', 'int64'])
# # Ajustar el modelo y predecir los outliers
# y_pred = lof.fit_predict(df_outliers)

# # El -1 indica un outlier, mientras que 1 indica un inlier
# df['outlier'] = y_pred

# # Filtrar los outliers
# outliers = df[df['outlier'] == -1]
# inliers = df[df['outlier'] == 1]

# # Contar el número total de muestras
# total_muestras = len(df)

# # Contar el número de outliers
# num_outliers = df['outlier'].value_counts().get(-1, 0)

# # Calcular el porcentaje de outliers
# porcentaje_outliers = (num_outliers / total_muestras) * 100

# print(f'Número total de muestras: {total_muestras}')
# print(f'Número de outliers: {num_outliers}')
# print(f'Porcentaje de outliers: {porcentaje_outliers:.2f}%')

# # Obtener los índices de los outliers
# outlier_indices = df[df['outlier'] == -1].index

# # Eliminar los outliers de X y y utilizando los índices
# X_cleaned = df.drop(index=outlier_indices).reset_index(drop=True)
# print(len(X_cleaned))

# ### 1.7. Análisis de correlación

matriz_correlaciones = df.corr()
plt.figure(figsize=(20, 16))
sns.heatmap(matriz_correlaciones, annot=True, fmt=".2f")
plt.title('Matriz de Correlación')
plt.show()

# Observaciones de correlación:
#  - Numerosas variables con correlacion de 1, lo que significa que ocupan los mismos valores, ya que aumentan paralelamente.
# 
#  Observaciones de datos:
# 
# Interpretación de colores: 
# - MB_TOTALES se mide en meses (correlación 1 respecto MB_MENSUALES)
# - Los usuarios que se crean recientemente una cuenta suelen estar muy activos
# - Cuantos más errores en el servicio(NUM_DIAS_BUNDLE), menos actividad (NUM_DIAS_ACTIVO)
# - [...]
# 
# Referencias:
# 
#  - Documentación vista en clase (Regresion_PrecioDiamantes.ipynb)
# 
#  - [Interpretación de correlación](https://www.cimec.es/coeficiente-correlacion-pearson/#:~:text=Un%20valor%20mayor%20que%200,una%20relaci%C3%B3n%20lineal%20positiva%20perfecta.)

#  Valores de coorelacion similar como (NUM_DESACTIVACIONES_FIJAS_POSPAGO y NUM_DESACTIVACIONES_FIJAS) y (NUM_LINEAS_TECNOLOGIA_DESCONOCIDA y NUM_SERVICIOS_POSPAGO), por lo que al tener estas los mismos valores se procede a la eliminacion de una de las 2

if 'NUM_DESACTIVACIONES_FIJAS' in df.columns:
    df = df.rename(columns={'NUM_DESACTIVACIONES_FIJAS_POSPAGO': 'DESACTIVACIONES_FIJAS_POSPAGO_INCL_DESACTIVACIONES_FIJAS'})
    df = df.drop(columns=['NUM_DESACTIVACIONES_FIJAS'])

if 'NUM_LINEAS_TECNOLOGIA_DESCONOCIDA' in df.columns:
    df = df.rename(columns={'NUM_SERVICIOS_POSPAGO': 'SERVICIOS_POSPAGO_INCL_LINEAS_TECNOLOGIA_DESCONOCIDA'})
    df = df.drop(columns=['NUM_LINEAS_TECNOLOGIA_DESCONOCIDA'])

# ## 2. Preparación de los datos para el modelado

# ### 2.1. Selección de variables de entrenamiento.

X = df.drop(['TARGET', 'MES'], axis=1)
y = df['TARGET']

# ### 2.2. Estandarización

columnas_num = X.select_dtypes(include=['float64', 'int64'])
scaler = StandardScaler()
x_scaled = scaler.fit_transform(columnas_num)
x_estandarizado = pd.DataFrame(x_scaled, columns=X.columns)
X = X.drop(columns=columnas_num.columns).reset_index(drop=True)
X = pd.concat([X, x_estandarizado], axis=1)
X.head()


