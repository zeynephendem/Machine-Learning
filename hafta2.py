#Model Performans Ölçütleri

import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

y_true=np.array([0,1,0,1,1,0,0,0])
y_pred=np.array([0,1,0,1,1,0,0,0])

accuracy=accuracy_score(y_true,y_pred)
precision=precision_score(y_true,y_pred)
recall=recall_score(y_true,y_pred)
f1=f1_score(y_true,y_pred)

karışıklık=confusion_matrix(y_true,y_pred)
specificty=karışıklık[0,0]/(karışıklık[0,0]+karışıklık[0,1])

print("Doğruluk:",accuracy)
print("Kesinlik:",precision)
print("Duyarlılık:",recall)
print("F1 Skoru:",f1)
print("Özgüllük:",specificty)
print("Karışıklık Matrisi:\n",karışıklık)


#ROC eğrisinin çizdirilmesi

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc

y_true=["Evet","Hayır","Evet","Hayır","Evet"]
y_pred=["Evet","Hayır","Hayır","Evet","Evet"]

y_true_binary=[1 if y=="Evet" else 0 for y in y_true]
y_pred_binary=[1 if y=="Evet" else 0 for y in y_pred]

fpr,tpr,thresholds=roc_curve(y_true_binary,y_pred_binary)
roc_auc=auc(fpr,tpr)

plt.figure()
plt.plot(fpr,tpr,color="blue",lw=2,label=f"ROC curve (AUC={roc_auc:.2f})")
plt.plot([0,1],[0,1],color="gray",linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()


#Örnek - 2
y_true=["Hayır", "Evet", "Hayır", "Evet", "Evet", "Hayır", "Hayır", "Hayır"]
y_pred=["Hayır", "Evet", "Hayır", "Evet", "Evet", "Hayır", "Hayır", "Hayır"]

y_true_binary = [1 if y == "Evet" else 0 for y in y_true]
y_pred_binary = [1 if y == "Evet" else 0 for y in y_pred]

fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_binary)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
