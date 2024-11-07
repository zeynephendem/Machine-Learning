import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#Veri Kümelerini Yükleme
egitim_data=pd.DataFrame({
    "MODEL":["X5","X3","X5","X3","X3","X3","X3","X3"],
    "CİNSİYET":["ERKEK","KADIN","ERKEK","ERKEK","ERKEK","KADIN","KADIN","ERKEK"],
    "YAŞ":[21,19,22,21,30,60,45,55],
    "MEMNUN":["HAYIR","EVET","HAYIR","EVET","EVET","HAYIR","HAYIR","HAYIR"]
})
test_data=pd.DataFrame({
    "MODEL":["X3","X5","X5","X3","X3"],
    "CİNSİYET":["ERKEK","ERKEK","KADIN","KADIN","KADIN"],
    "YAŞ":[25,45,66,36,25],
    "MEMNUN":["HAYIR","EVET","EVET","HAYIR","EVET"]
})

#Kategorik Verileri Sayısal Verilere Dönüştürme
egitim_data["CİNSİYET"]=egitim_data["CİNSİYET"].map({"ERKEK":0,"KADIN":1})
test_data["CİNSİYET"]=test_data["CİNSİYET"].map({"ERKEK":0,"KADIN":1})

egitim_data["MODEL"]=egitim_data["MODEL"].map({"X3":0,"X5":1})
test_data["MODEL"]=test_data["MODEL"].map({"X3":0,"X5":1})

egitim_data["MEMNUN"]=egitim_data["MEMNUN"].map({"HAYIR":0,"EVET":1})
test_data["MEMNUN"]=test_data["MEMNUN"].map({"HAYIR":0,"EVET":1})

#Eğitim ve Test Veri Kümelerini Ayırma
X_train=egitim_data[["MODEL","CİNSİYET","YAŞ"]]
y_train=egitim_data["MEMNUN"]

X_test=test_data[["MODEL","CİNSİYET","YAŞ"]]
y_test=test_data["MEMNUN"]

#Decision Tree (C4.5) Sınıflandırıcı
model=DecisionTreeClassifier(criterion="entropy")
model.fit(X_train,y_train)

#Performans Değerlendirme (Eğitim veri kümesi)
y_train_pred=model.predict(X_train)
train_accuracy=accuracy_score(y_train,y_train_pred)
print(f"Eğitim Performansı (Doğruluk):{train_accuracy:.2f}")

#Performans Değerlendirme (TEST veri kümesi)
y_test_pred=model.predict(X_test)
test_accuracy=accuracy_score(y_test,y_test_pred)
print(f"Test Performansı (Doğruluk):{test_accuracy:.2f}")

#Karar Ağacını Görselleştirme
plt.figure(figsize=[5,5])

plot_tree(model,filled=True,feature_names=["MODEL","CİNSİYET","YAŞ"]),

class_names=["HAYIR","EVET"]
plt.title("Karar Ağacı Şeması")

plt.show()
