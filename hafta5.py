#Bayes Sınıflandırıcılar

#Uygulama - 1

from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

egitim_verisi=[
    ["Orta", "Yaşlı", "Erkek", "Evet"],
    ["İlk", "Genç", "Erkek", "Hayır"],
    ["Yüksek", "Orta", "Kadın", "Hayır"],
    ["Orta", "Orta", "Erkek", "Evet"],
    ["İlk", "Orta", "Erkek", "Evet"],
    ["Yüksek", "Yaşlı", "Kadın", "Evet"],
    ["İlk", "Genç", "Kadın", "Hayır"],
    ["Orta", "Orta", "Kadın", "Evet"]
    ]

test_verisi=[["Orta","Orta","Kadın"]]

encoder=OrdinalEncoder()
egitim_verisi_encoded=encoder.fit_transform([row[:-1] for row in egitim_verisi])
egitim_etiketleri=[row[-1] for row in egitim_verisi]

model=CategoricalNB()
model.fit(egitim_verisi_encoded,egitim_etiketleri)

test_verisi_encoded=encoder.transform(test_verisi)

tahmin=model.predict(test_verisi_encoded)

print("Test verisi için tahmin edilen kabul durumu:", tahmin[0])

#Uygulama - 2

from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

egitim_verisi = [
    ["düşük", "yok", "az","hayır"],
    ["düşük", "yok", "çok", "hayır"],
    ["yüksek", "var", "çok", "evet"],
    ["yüksek", "var", "az", "evet"],
    ["yüksek", "yok", "az", "evet"],
    ["yüksek", "var", "çok", "evet"],
    ["yüksek", "var", "az", "hayır"],
    ["yüksek", "yok", "az", "evet"]
    ]

test_verisi = [["düşük", "var", "çok"]]

encoder = OrdinalEncoder()
egitim_verisi_encoded = encoder.fit_transform([satır[:-1] for satır in
egitim_verisi]) 
model = CategoricalNB()
model.fit(egitim_verisi_encoded, egitim_etiketleri)

test_verisi_encoded = encoder.transform(test_verisi)

tahmin = model.predict(test_verisi_encoded)
print("Test verisi için tahmin edilen GRİP durumu:", tahmin[0])
