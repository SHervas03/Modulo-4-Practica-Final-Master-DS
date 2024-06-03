# # Practica Final
# Modulo 4 - Máster Data Science y Business Analytics
# Sergio Hervás Aragón

# ### Librerías

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, precision_score, recall_score
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import missingno as msno
from sklearn.datasets import make_blobs
from sklearn.metrics import roc_curve


# ## 1. EDA

# ### 1.1. Carga de datos

# Declaración de una variable, la que se encargara de guardar nuestros registros
df = []
file = './caso_final_small_20k_con_mes.csv'
df = pd.read_csv(file, sep=',')
df

# ### 1.2. Descripción y Análisis

df.info()

# Observaciones:
# 
# - Columnas con valores nulos
# - Diferentes tipos de datos (float64, int64)
# 

df.describe()

msno.matrix(df)

# Observaciones:
# 
# Mostramos una matriz de calor para visualizar las variables con mayor preencia de valores perdidos, donde las lineas blancas indican los valores faltantes en cada columna.
# 
# Referencias:
# 
#  - [guia-sobre-tecnicas-de-imputacion-de-datos-con-python](https://cesarquezadab.com/2021/09/19/guia-sobre-tecnicas-de-imputacion-de-datos-con-python/)
# 

# ### 1.3. Distribucion de la variable objetivo

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='TARGET')
plt.title('Distribución de Supervivencia')
plt.xlabel('Supervivencia')
plt.ylabel('Frecuencia')
plt.grid(False)
plt.show()

# ### 1.4. Tratamiento de nulos mediante imputación

imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
msno.matrix(df)

# Observaciones:
# 
#  Uso de KNNInputer, la cual consiste en asignar a cada dato nulo un valor obtenido a partir de la información disponible de los 5 vecinos más cercanos o parecidos a este.
# 
# Referencias
# 
#  - [guia-sobre-tecnicas-de-imputacion-de-datos-con-python](https://cesarquezadab.com/2021/09/19/guia-sobre-tecnicas-de-imputacion-de-datos-con-python/)
# 
#  - [Impacto de estrategias](https://rephip.unr.edu.ar/server/api/core/bitstreams/b8e75f0e-1df0-462f-b309-43eeaf0fdcbc/content#:~:text=KNN%20es%20un%20m%C3%A9todo%20eficiente,este%20(donantes%20o%20vecinos).)
# 
#  - Documentación vista en clase (Clasificación_Titanic.ipynb)

# ### 1.4. Outliers

clf = LocalOutlierFactor(n_neighbors=8, contamination='auto')
y_pred = clf.fit_predict(df)
n_outliers = sum(y_pred==-1)
n_total = len(y_pred)
X_scores = clf.negative_outlier_factor_
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
print(u'El número de outliers detectados es de {} de un total de {}'.format(n_outliers, n_total))

# Referencias
# 
#  - [LocalOutlierFactor](https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection)

# ### 1.5. Análisis de correlación

matriz_correlaciones = df.corr()
plt.figure(figsize=(20, 16))
sns.heatmap(matriz_correlaciones, annot=True, fmt=".2f")
plt.title('Matriz de Correlación')
plt.show()

# Observaciones:
# 
# Interpretación de colores: 
#  - MB_TOTALES se mide en meses (correlación 1 respecto MB_MENSUALES)
#  - Los usuarios que se crean recientemente una cuenta suelen estar muy activos
#  - Cuantos más errores en el servicio(NUM_DIAS_BUNDLE), menos actividad (NUM_DIAS_ACTIVO)
#  - [...]
# 
# Referencias:
# 
#  - Documentación vista en clase (Regresion_PrecioDiamantes.ipynb)
# 
#  - [Interpretación de correlación](https://www.cimec.es/coeficiente-correlacion-pearson/#:~:text=Un%20valor%20mayor%20que%200,una%20relaci%C3%B3n%20lineal%20positiva%20perfecta.)
#  

# ## 2. Preparación de los datos para el modelado

# ### 2.1. Selección de variables de entrenamiento.

x = df.drop(['TARGET', 'MES'], axis=1)
y = df['TARGET']

# ### 2.2. Estandarización

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_estandarizado = pd.DataFrame(x_scaled, columns=x.columns)
x_estandarizado

# ### 2.3. Division del dataset

x_train, x_test, y_train, y_test = train_test_split(x_estandarizado, y, test_size=0.2, random_state=0)

print(u'Dimensiones en train \n-x:{}\n-y:{}'.format(x_train.shape, y_train.shape))
print(u'Dimensiones en test \n-x:{}\n-y:{}'.format(x_test.shape, y_test.shape))

# ## 3. Comparación del rendimiento de varios modelos

# ### 3.1. Regresión logística

lr = LogisticRegression()
lr.fit(x_train, y_train)
y_test_pred_lr = lr.predict(x_test)
y_test_prob_lr = lr.predict_proba(x_test)

fpr, tpr, thrs = roc_curve(y_test, y_test_prob_lr[:, 1])
plt.figure(figsize=(12,12))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], "r--")
plt.title("ROC")
plt.xlabel("Falsos Positivos")
plt.ylabel("Verdaderos Positivos")
plt.show()

auc = roc_auc_score(y_test, y_test_prob_lr[:, 1])
print("- Precision:", round(precision_score(y_test, y_test_pred_lr),2))
print("- Recall:", recall_score(y_test, y_test_pred_lr))
print("- Fscore:", round(f1_score(y_test, y_test_pred_lr),2))
print("- AUC:", round(auc,2))


# ### 3.2. Modelo ensamblado

# [Bagging](http://eio.usc.es/pub/mte/descargas/ProyectosFinMaster/Proyecto_1686.pdf)

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
y_test_pred_rfc = rfc.predict(x_test)
y_test_prob_rfc = rfc.predict_proba(x_test)

fpr, tpr, thrs = roc_curve(y_test, y_test_prob_rfc[:, 1])
plt.figure(figsize=(12,12))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], "r--")
plt.title("ROC")
plt.xlabel("Falsos Positivos")
plt.ylabel("Verdaderos Positivos")
plt.show()

auc = roc_auc_score(y_test, y_test_prob_rfc[:, 1])
print("- Precision:", round(precision_score(y_test, y_test_pred_rfc),2))
print("- Recall:", recall_score(y_test, y_test_pred_rfc))
print("- Fscore:", round(f1_score(y_test, y_test_pred_rfc),2))
print("- AUC:", round(auc,2))


# ### 3.3. Red neuronal (MLP)

mlp = MLPClassifier()
mlp.fit(x_train, y_train)
y_test_pred_mlp = mlp.predict(x_test)
y_test_prob_mlp = mlp.predict_proba(x_test)

fpr, tpr, thrs = roc_curve(y_test, y_test_prob_mlp[:, 1])
plt.figure(figsize=(12,12))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], "r--")
plt.title("ROC")
plt.xlabel("Falsos Positivos")
plt.ylabel("Verdaderos Positivos")
plt.show()

auc = roc_auc_score(y_test, y_test_prob_mlp[:, 1])
print("- Precision:", round(precision_score(y_test, y_test_pred_mlp),2))
print("- Recall:", recall_score(y_test, y_test_pred_mlp))
print("- Fscore:", round(f1_score(y_test, y_test_pred_mlp),2))
print("- AUC:", round(auc,2))

# ## 4. Segmentación de clientes

km = KMeans(n_clusters=2, random_state=0)
km = km.fit_predict(x_estandarizado)
km

rfc_cluster = LogisticRegression()
rfc_cluster.fit

# [KMeans clustering](https://www.kaggle.com/code/micheldc55/introduccion-al-clustering-con-python-y-sklearn)
# 
# [El algoritmo k-means](https://www.unioviedo.es/compnum/laboratorios_py/kmeans/kmeans)

# ## 5. Tratamiento y análisis de la columna Mes.

# Calculamos los cuantiles y el IQR
Q1 = df['MES'].quantile(0.25)
Q3 = df['MES'].quantile(0.75)
IQR = Q3 - Q1

# Calculamos los límites para identificar outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identificamos outliers por encima y por debajo de los límites
outliers_above = df['MES'][df['MES'] > upper_bound]
outliers_below = df['MES'][df['MES'] < lower_bound]

print("Número de outliers por encima:", outliers_above.shape[0])
print("Número de outliers por debajo:", outliers_below.shape[0])

# Boxplot vertical
plt.figure(figsize=(4, 6))
sns.boxplot(y=df['MES'], color='skyblue', orient='v')
plt.ylabel('Precio')
plt.title('Boxplot Y')
plt.show()

df['MES'] = pd.to_datetime(df['MES'], format='%Y%m')

df['MES'].info()

df['MES'].describe()

# - [pandas.to_datetime](https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html)
# 
# - [Float Python](https://ellibrodepython.com/float-python)
# 
#  - Documentación vista en clase (Regresion_PrecioDiamantes.ipynb)


