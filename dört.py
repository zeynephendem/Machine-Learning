import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#Veriyi NumPy dizilerini dönüştürme (Evet=1, Hayır=0)
y_true=np.array([0,1,0,1,1,0,0,0])
y_pred=np.array([0,1,0,1,1,0,0,0])

#Metrikleri hesaplama
accuracy=accuracy_score(y_true,y_pred)
precision=precision_score(y_true,y_pred)
recall=recall_score(y_true,y_pred)
f1=f1_score(y_true,y_pred)
specificity=confusion_matrix(y_true,y_pred)[0,0]/(confusion_matrix(y_true,y_pred)[0,0]+confusion_matrix(y_true,y_pred)[0,1])

#Sonuçları yazdırma
print("Doğruluk:",accuracy)

print("Kesinlik:",precision)

print("Duyarlılık:",recall)

print("F1 Skoru:",f1)

print("Özgüllük:",specificity)

print("Karışıklık Matrisi:\n", confusion_matrix(y_true, y_pred))
