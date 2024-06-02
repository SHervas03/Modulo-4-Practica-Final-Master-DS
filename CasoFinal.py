# # Practica Final
# Modulo 4 - Máster Data Science y Business Analytics
# Sergio Hervás Aragón

# ### Librerías

# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Preprocesado
# ==============================================================================
from sklearn.impute import KNNImputer

import missingno as msno # librería para tratamiento de datos perdidos

# ### Carga de datos

def read_csv(file, sep):
    # Declaración de una variable, la que se encargara de guardar nuestros registros
    df = []
    try:
        # Lectura de nuestro fichero
        df = pd.read_csv(file, sep=sep)
    except FileNotFoundError:
        # Excepción en caso de no ser el archivo encontrado por la ruta ofrecida
        raise ValueError('Archivo no encontrado')
    except Exception as e:
        # Excepción genérica en caso de que haya otro problema
        raise ValueError(f'Error: {e}')
    finally:
        print('Proceso de lectura finalizado')
    return df

# ### Descripción y Análisis

def description_and_analysis(df):
    print(f'\nDescripción de la tabla de la tabla:\n {df.describe()}')
    print('\nInformación de la tabla de la tabla:\n ')
    print(f'{df.info()}')
    msno.matrix(df)
    plt.title('Valores faltantes en cada columna')


# ### Tratamiento de nulos mediante imputación

def null_value_treatment(df):
    # https://cesarquezadab.com/2021/09/19/guia-sobre-tecnicas-de-imputacion-de-datos-con-python/
    # Documentación vista en clase (Clasificación_Titanic.ipynb)
    imputer = KNNImputer(n_neighbors=5)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# ### Outliers



# ## Analisis exploratorio

# ### Conteo de Nulos

def count_off_nulls(df):
    print('\nConteo de nulos por columnas:')
    # Variable que nos servira para verificar si hay o no valores nulos
    null_validator = True
    # Bucle donde se recorren las columnas una a una
    for column in df.columns:
        null_count = df[column].isnull().sum()
        # Si existen nulos, imprimir el resultaso de la columna con nulos
        if null_count != 0:
            null_validator = False
            print(f' - {column} | tiene {null_count} nulos')
    # Comprobador de valores nulos final
    if not null_validator:
        print('\nHay valores nulos en el DataFrame')
        null_validator = True
    else:
        print(' - No hay nulos en el DataFrame')

# ### Reemplazo de nulos por 0:
# Tras la exploración de los datos con los que vamos a trabajar, al ser de tipo numérico, se opta por el reemplazo de estos por un valor numérico, de tal manera que no se borre ningún dato, pero que a su vez no cuente para el análisis

def replace_null_with_zero(df):
    print('\nReemplazo de NULL´s por 0')
    # Reemplazo de valores nulos por 0
    df.fillna(0, inplace=True)

# ### Resumen estadístico de variables

def variable_summary(df):
    print(df.describe())

# ### Transformación de datos de float a date
# 
# Tras la visualización de los datos, se opta por un parseo de la columna 'MES' a date para un mejor tratamiento de datos

def parse_column_year(df):
    df['MES'] = df['MES'].astype(str)
    df['MES'] = pd.to_datetime(df['MES'], format='%Y%m')
    return df

if __name__ == "__main__":
    df = read_csv(file = './caso_final_small_20k_con_mes.csv', sep = ',')
    description_and_analysis(df)
    null_value_treatment(df)
    # count_off_nulls(df)
    # replace_null_with_zero(df)
    # count_off_nulls(df)
    # df = parse_column_year(df)
    # column_type_display(df)
    # variable_summary(df)


# # Bibliografía
# 
# ### column_parsing:
# 
#  - [Float Python](https://ellibrodepython.com/float-python)
#  
#  - [Evitar notación científica](https://es.stackoverflow.com/questions/533263/como-imprimir-cifras-completas-en-un-dataframe-que-no-aparezca-el)
# 
#  - [pandas.to_datetime](https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html)
# 


