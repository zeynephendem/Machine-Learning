#Çoklu Regresyon

#Uygulama - 1

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

veri=pd.DataFrame({
    "yıl":list(range(2014,2022)),
    "satış":[265,280,259,295,270,265,255,276],
    "fiyat":[25,25,30,30,35,36,38,41],
    "rakip":[30,32,32,35,35,35,35,38]})

print(veri)

model = LinearRegression()
x = veri[['fiyat', 'rakip']]
y = veri['satış']
model.fit(x, y)

print("Katsayılar:", model.coef_)
print("Kesme terimi:", model.intercept_)

öngörü = model.predict(x)

mse = mean_squared_error(y, öngörü)
print("MSE:", mse)

#Uygulama - 2

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

wine = load_wine()
x,y=wine.data, wine.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

print("Katsayılar:", model.coef_)
print("Kesme terimi:", model.intercept_)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

print("Eğitim için MSE:", mean_squared_error(y_train, y_train_pred))
print("Test için MSE:", mean_squared_error(y_test, y_test_pred))
