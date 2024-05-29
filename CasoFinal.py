# # Practica Final
# Modulo 4 - Máster Data Science y Business Analytics
# Sergio Hervás Aragón

# ### Importaciones

import pandas as pd
import sys

# ### Lectura de datos

def read_csv(file):
    # Declaración de una variable, la que se encargara de guardar nuestros registros
    df = []
    try:
        # Lectura de nuestro fichero
        df = pd.read_csv(file, sep=';')
    except FileNotFoundError:
        # Excepción en caso de no ser el archivo encontrado por la ruta ofrecida
        print('Error: Archivo no encontrado')
        # Finalizador de programa en caso de saltar la excepción
        sys.exit(1)
    except pd.errors.ParserError:
        # Excepción en caso de tener mal indicado el separador a usar
        print('Error: Archivo erroneamente parseado')
        # Finalizador de programa en caso de saltar la excepción
        sys.exit(1)
    except Exception as e:
        # Excepción genérica en caso de que haya otro problema
        print(f'Error: {e}')
        sys.exit(1)
    finally:
        print('Proceso de lectura finalizado')
    return df

# ### Analisis exploratorio

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

def replace_null_with_zero(df):
    print('\nReemplazo de NULL´s por 0')
    # Reemplazo de valores nulos por 0
    df.fillna(0, inplace=True)

def column_type_display(df):
    print(f'\ndtypes de las columnas: \n{df.dtypes}')

def column_parsing(df):
    # pd.options.display.float_format = '{:.3f}'.format
    for column in df.columns:
        if df[column].dtypes == 'object':
            try:
                df[column] = df[column].astype('float64')
            except ValueError:
                df[column] = df[column].str.replace(',','')
                try:
                    df[column] = df[column].astype('float64')
                except ValueError:
                    df[column] = df[column].str.replace('.','')
                    try:
                        df[column] = df[column].astype('float64')
                    except ValueError:
                        print('Hay columnas que no pueden ser parseadas')
    print("\nProceso de parseo finalizado")


if __name__ == "__main__":
    df = read_csv('./caso_final_small_20k_con_mes.csv')
    count_off_nulls(df)
    replace_null_with_zero(df)
    count_off_nulls(df)
    column_type_display(df)
    column_parsing(df)
    column_type_display(df)


# # Bibliografía
# 
# ### column_parsing:
# 
#  - [Float Python](https://ellibrodepython.com/float-python)
#  
#  - [Evitar notación científica](https://es.stackoverflow.com/questions/533263/como-imprimir-cifras-completas-en-un-dataframe-que-no-aparezca-el)
# 


