#Performans Geliştirme Yöntemleri

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold,cross_val_score
from sklearn.ensemble import RandomForestClassifier

iris=load_iris()
x=iris.data
y=iris.target

clf=RandomForestClassifier(n_estimators=100,random_state=42)
cv=KFold(n_splits=10,shuffle=True,random_state=42)
scores=cross_val_score(clf,x,y,cv=cv)
fold_numbers=np.arange(1,11)
df=pd.DataFrame({"Fold":fold_numbers,"Accuracy":scores})

average_accuracy=np.mean(scores)
avg_row = pd.DataFrame({"Fold": ["Ortalama"], "Accuracy":
[average_accuracy]})
result_table = pd.concat([df, avg_row], ignore_index=True)

print(result_table)

#Doğrusal Regresyon

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

veri = pd.DataFrame({
"araç_sayısı":[14317,15096,16090,17033,17939,18829,19994,21090,22219,22866,23157],
"ölü_sayısı": [4.324,4.045,3.835,3.750,3.685,3.524,7.530,7.300,7.427,6.675,5.473]
})

print(veri)

plt.scatter(veri["araç_sayısı"],veri["ölü_sayısı"])
plt.title("Ölüm sayıları ile araç sayısı arasındaki ilişki")
plt.xlabel("Araç Sayısı")
plt.ylabel("Ölüm Sayısı")
plt.show()

x=veri[["araç_sayısı"]]
y=veri["ölü_sayısı"]
model=LinearRegression().fit(x,y)

print(f"Katsayılar: {model.coef_[0]}(b-eğim),{model.intercept_}(a-kesme terimi)")

yeni_arac_sayısı = 23500
tahmin = model.predict(pd.DataFrame({"araç_sayısı": [yeni_arac_sayısı]}))

print(f"Araç sayısı {yeni_arac_sayısı} için tahmin edilen ölüm sayısı:{tahmin[0]}")

tahminler = model.predict(x)
mse = mean_squared_error(y, tahminler)
print(f"MSE (Ortalama Kare Hata): {mse}")

      
