# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
import pandas as pd
import os 
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (precision_score, balanced_accuracy_score,
                              recall_score, f1_score)
from sklearn.metrics import confusion_matrix
import gzip
import pickle
import os
import json

def cargar_datos():

    df_train=pd.read_csv("files/input/train_data.csv.zip")
    df_test=pd.read_csv("files/input/test_data.csv.zip")
    return df_train, df_test

#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
def limpieza(df):
    df=df.copy()
    df=df.rename(columns={"default payment next month":"default"})
    df=df.drop(columns=["ID"], errors="ignore")
    df=df.dropna()
    df=df[(df["EDUCATION"]>0) & (df["MARRIAGE"]>0)]
    df.loc[df["EDUCATION"]>=4, "EDUCATION"]=4
    
    return df
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
def separar_datos(base):
    base=base.copy()
    x=base
    y=base.pop("default")
    return x,y
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
def hacer_pipeline(estimador):
    columnas_categoricas=["SEX","EDUCATION","MARRIAGE"]

    preproceso=ColumnTransformer(
        transformers=[
            ("ohe",
             OneHotEncoder(handle_unknown="ignore"),
             columnas_categoricas),
        ],
        remainder=StandardScaler()
    )

    pipeline=Pipeline([
        ("preprocessor", preproceso),
        ("feature_selection", SelectKBest(score_func=f_classif)),
        ("pca", PCA()),
        ("classifier", MLPClassifier(max_iter=15000, random_state=17))
    ])

    return pipeline
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
def optimizar_modelo(pipeline, x_train, y_train):
    param_grid = {
        'pca__n_components': [None],
        'feature_selection__k': [20],
        'classifier__hidden_layer_sizes': [(50, 30, 40, 60)],
        'classifier__alpha': [0.26],
        'classifier__learning_rate_init': [0.001]
    }

    grid_search=GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=10,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train, y_train)

    return grid_search
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
def guardar_modelo(modelo):
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(modelo, f)
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
def calcular_metricas(modelo, X, y, nombre_dataset: str):
    y_pred = modelo.predict(X)
    metricas = {
        "type": "metrics",
        "dataset": nombre_dataset,
        "precision": round(precision_score(y, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y, y_pred), 4),
        "recall": round(recall_score(y, y_pred), 4),
        "f1_score": round(f1_score(y, y_pred), 4)
    }
    return y_pred, metricas
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
def calcular_matriz_confusion(y_true, y_pred, nombre_dataset: str):
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": nombre_dataset,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])}
    }

def guardar_registros(registros, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for item in registros:
            f.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    # Cargar datos
    df_train, df_test=cargar_datos()

    # Limpieza de datos
    df_train=limpieza(df_train)
    df_test=limpieza(df_test)

    # Separar datos en x e y
    x_train, y_train=separar_datos(df_train)
    x_test, y_test=separar_datos(df_test)

    # Crear pipeline
    estimador=SVC()
    pipeline=hacer_pipeline(estimador)

    # Optimizar modelo
    modelo_optimizado=optimizar_modelo(pipeline, x_train, y_train)

    # Guardar modelo
    guardar_modelo(modelo_optimizado)

    # Calcular metricas para train y test
    y_pred_train, metricas_train = calcular_metricas(modelo_optimizado, x_train, y_train, "train")
    y_pred_test, metricas_test = calcular_metricas(modelo_optimizado, x_test, y_test, "test")

    # Calcular matrices de confusion para train y test
    cm_train = calcular_matriz_confusion(y_train, y_pred_train, "train")
    cm_test = calcular_matriz_confusion(y_test, y_pred_test, "test")

    # Guardar metricas y matrices de confusion
    registros = [metricas_train, metricas_test, cm_train, cm_test]
    guardar_registros(registros, "files/output/metrics.json")