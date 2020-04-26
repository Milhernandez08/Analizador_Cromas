import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datos = pd.read_csv('DATASET.csv')
datos = datos.replace(np.nan, "0")

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

# df = pd.DataFrame()

X = datos[['Blanco','Amarillo','Cafe','Rojiso','Anaranjado','Lineas','Forma_n','Contorno (','Negro','Gris','Violeta','Verde','Azul','Dientes_Caballo']]
y = datos['RES']

# from sklearn.model_selection import train_test_split
from ValidadcionCruzada import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=500, alpha=0.0001, solver='adam', random_state=21, tol=0.000000001)
#mlp = MLPClassifier(hidden_layer_sizes=(5,5,5,5), max_iter=5000)

mlp.fit(X_train, y_train)
predicciones = mlp.predict(X_test)

#Grafica
# plt.scatter(X_train,y_train)
# plt.scatter(X_test,predicciones)

# print("Tamaño de X_train", len(X_train), "Tamaño de y_train: ", len(y_train))

# from sklearn.metrics import mean_squared_error
# print("Error: ", mean_squared_error(y_test, predicciones))

from sklearn.metrics import classification_report
print(classification_report(y_test, predicciones))

from sklearn.externals import joblib
joblib.dump(mlp, 'RED_N.pkl')