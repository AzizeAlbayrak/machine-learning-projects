import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


veriler = pd.read_csv("diabetes_clean.csv")
print(veriler.head()) #ilk 5 satır veriyi gösterir.

print(veriler.shape) #satır kolon sayısı
print(veriler.describe()) #caunt kaç veri var mean=ort 
#veri dağılımını anlarsın, aykırı değer var mı, ölçek farkı var mı kontrol yapmak için describe()

print(veriler.dtypes)
print("---------------Boş veriler-----------")
print(veriler.isnull().sum())


print("---------------Bağlantı-------------")
print(veriler.corr() ['Outcome'])

veriler = veriler.drop("Unnamed: 0", axis=1)
print(veriler.head())



X = veriler.iloc[:, :-1] #1den baila son kolonu alma
Y = veriler.iloc[:, -1 ]
print(Y)

from sklearn.model_selection import train_test_split

x_train,x_test, y_train, y_test = train_test_split(X,Y , test_size = 0.33, random_state= 0)


#olcekleme 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


#Logistik Regresyon
#Solver = optimizasyon algoritması en iyi parametreleri nasıl arayacağını belirler
#liblinear  küçük veri setlerinde iyi
#c= overfit underfit dengeler küçükse model basit
#penalty ceza terimi büyük katsayılar kullanmasın diye 
#L2 katsayıları küçültür L1 bazı katsayıları 0a indirir 

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
parametreler = {"C":[0.01,0.1,1,10],"penalty":["l1","l2"],"solver":["liblinear"]}

#max iter modelin daha iyi öğrenmesi için arttırıyoruz. Optimizasyon alg.
#en iyi ağırlıkları bulmak için tekrar tekrar hesaplama yapar
grid = GridSearchCV(LogisticRegression(max_iter=1000), parametreler, cv=5)
grid.fit(X_train,y_train)
print("En iyi parametre:", grid.best_params_)

#en iyi modeli al
model = grid.best_estimator_

#tahmin yap
y_pred = model.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(" ")
print("----------Logistik regresyon İçin-----------")
print(cm)

print("Accuracy:", accuracy_score(y_test, y_pred))


#precision , recall , f1 score
print(classification_report(y_test, y_pred))

#KNN
print("\n----------KNN  İçin-----------")
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn.fit(x_train,y_train)

y_pred =knn.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


#SVM
print("\n----------SVM İçin-----------")
from sklearn.svm import SVC
svc = SVC(kernel = 'linear')
svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


#Naive Bayes
print("\n----------Naive Bayes İçin-----------")
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



#decision tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


parametreler = {
    "max_depth":[2,3,4,5,6,7,8],
    "min_samples_split":[2,5,10],
    "min_samples_leaf":[1,2,4]
}
print("\nEN IYI PARAMETRELER:" , grid.best_params_)

model = grid.best_estimator_


grid = GridSearchCV(
    DecisionTreeClassifier(random_state=0),
    parametreler,
    cv=5,
    scoring="accuracy")

grid.fit(x_train,y_train)


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


dtc = DecisionTreeClassifier(criterion='entropy' , random_state=0 , max_depth=5, min_samples_leaf=2, min_samples_split=2)  

dtc.fit(x_train,y_train)
y_pred = dtc.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(" ")
print('----------Decision Tree Entropy İçin-------------')
print(cm)
      
print(classification_report(y_test, y_pred))


dtc = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=2, min_samples_leaf=2, min_samples_split=2)

dtc.fit(x_train,y_train)
y_pred = dtc.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print('-------------Decision Tree Gini İçin-------------')
print(cm)
      
print(classification_report(y_test, y_pred))


#Random Forest
print("\n---------------Random Forest İçin --------------")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

parametreler ={
    "n_estimators": [100,200],
    "criterion": ['gini', 'entropy'],
    "max_depth" : [None , 2 ,5],
    "min_samples_leaf": [1,2]
    }

grid = GridSearchCV(RandomForestClassifier(), parametreler, cv = 3, scoring= 'accuracy')
grid.fit(x_train, y_train)
print("En İyi Parametreler:" , grid.best_params_)

model = grid.best_estimator_
y_pred = model.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))






