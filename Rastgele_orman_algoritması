from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report

#İris veri kümesi yüklenir
iris = load_iris()
X = iris.data
y = iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)

#Random Forest sınıflandırıcı modeli oluşturulur
model = RandomForestClassifier(n_estimators=100, random_state=42)

#Model eğitilir
model.fit(X_train, y_train)

#Test kümesi üzerinde öngörü yapılır
y_öngörü = model.predict(X_test)

#Modelin performansını değerlendirin
doğruluk = accuracy_score(y_test, y_öngörü)
print("Test verisiyle doğruluk:", doğruluk)


