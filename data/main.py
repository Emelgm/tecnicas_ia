# librerías necesarias
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.tree import  plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def run():
    # carga del dataset
    url = '/mnt/c/Users/USUARIO/Documents/tmp/unir/tecnicas_IA/tecnicas_ia/data/Laboratorio_dataset_car.csv'
    df_raw = pd.read_csv(url, header=None, names=["dataset"])
    print('*'*65)
    print('Dataset crudo')
    print('*'*65)
    print('\n',df_raw.head(),'\n')

    # limpieza
    # separar por ";"
    df = list(df_raw["dataset"].apply(lambda x:x.split(';')))
    # extraer el primer registro como cabecera
    header = df.pop(0)
    # crear un nuevo DF
    df_clean = pd.DataFrame(data=df, columns=header)
    print('*'*65)
    print('Dataset ordenado')
    print('*'*65)
    print('\n',df_clean.tail(),'\n')

    # caracterización del dataset
    # Cantidad de instancias en total
    print('*'*65)
    print('Caracterización de Dataset')
    print('*'*65)
    print('-'*45)
    total_instancias = len(df_clean)
    print(f'Total de instancias: {total_instancias}')
    # total instancias por clase
    print('-'*45)
    total_por_clase = df_clean.count()
    print(f'Total de instancias para cada clase: \n{total_por_clase}')
    # cantidad de atributos de entrada
    print('-'*45)
    cantidad_atributos = len(df_clean.columns)
    print(f'Cantidad de atributos de entrada: {cantidad_atributos}')
    # tipos de datos
    print('-'*45)
    print(f'Tipo de cada atributo: \n{df_clean.dtypes}')
    # exploración de datos nulos o desconocidos
    print('-'*45)
    datos_nulos = df_clean.isnull().sum().sum()
    print(f'Cantidad de datos desconocidos: {datos_nulos}\n')

    # exploración de datos
    print(df_clean["class"].value_counts())
    # generar grafico
    #sns.histplot(df_clean["class"])


if __name__=='__main__':
    run()