import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error,r2_score



veriler = pd.read_csv('Medical Cost Personal Datasets.csv')
print(veriler.head())

print(veriler.shape) #satır kolon sayısı

print(veriler.describe()) #caunt kaç veri var mean=ort 
#veri dağılımını anlarsın, aykırı değer var mı, ölçek farkı var mı kontrol yapmak için describe()

print(veriler.dtypes)
print(veriler.isnull().sum())

#encoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

veriler["sex"] = le.fit_transform(veriler["sex"])
veriler["smoker"] = le.fit_transform(veriler["smoker"])
print(veriler)


""" Yerlerini karıştırma ihtimaline karşı bunu kullanmamak daha sağlıklı olabilir
veriler.sex = le.fit_transform(veriler.iloc[:,1]) #fit inceler trasform sayısala çevirir
veriler.smoker = le.fit_transform(veriler.iloc[:, 4])
"""

#onehotencoder
ohe = preprocessing.OneHotEncoder()
region = veriler.iloc[:,[5]]
region = ohe.fit_transform(region).toarray()
print(region)

list(range(1338))

sonuc = pd.DataFrame(data=region, index = range(1338), columns=['southwest' ,'southeast', 'northwest' ,'northeast'])
print(sonuc)

veriler = pd.concat([veriler,sonuc], axis=1)
print(veriler.head())

#hala region sütunu var onu yok etmemiz lazım
veriler = veriler.drop("region" , axis =1)


print("---------------------------BAĞLANTI----------------------------")
print(veriler.corr() ['charges']) #bağımlı bağımsız arası basit bağlantı

#burada iloc ile belirtmek istersen chargesi son kolona taşıman daha iyi olur
X = veriler.drop("charges", axis=1)
Y = veriler["charges"]

X_sm = sm.add_constant(X) #yeni  bir kolon ekler o yüzden const var bias ekliyoruz aslında
model = sm.OLS(Y, X_sm).fit() #OLS = Ordinary Least Squares lineer reg. algoritması en iyi model için
#Tahmin ile gerçek değer arasındaki hatayı minimize etmek
print(model.summary()) #tam analiz 


X = X.drop('sex' , axis = 1)
X_sm = sm.add_constant(X)

model = sm.OLS(Y, X_sm).fit()
print(model.summary())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state= 0)


#lineer regresyon   
print(" ")
print("----------Lineer Regresyon İçin-----------")


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
parametreler = {"fit_intercept": [True, False] , "positive": [True , False]}

grid = GridSearchCV(LinearRegression(), parametreler , cv =5 , scoring="r2")
grid.fit(x_train,y_train)
print("EN IYI PARAMETRELER:" , grid.best_params_)

model = grid.best_estimator_

y_pred = model.predict(x_test)

print(" ")
#model başarısı
print("MSE:", mean_squared_error(y_test, y_pred)) #Modelin toplam hata büyüklüğü
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred))) #ortalama hata
print("R2:", r2_score(y_test, y_pred)) 


#Decision Tree
print("\n----------Decision Tree İçin-----------")

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

parametreler = {
    "max_depth":[2,3,4,5,6,7,8],
    "min_samples_split":[2,5,7,10],
    "min_samples_leaf":[1,2,4],
    "random_state": [0,42]
}

grid = GridSearchCV(DecisionTreeRegressor(), parametreler, cv = 5 , scoring = "r2")
grid.fit(x_train, y_train)
print("En iyi Parametreler ", grid.best_params_)

model1 = grid.best_estimator_
y_pred = model1.predict(x_test)

print("\nMSE:" , mean_squared_error(y_test, y_pred))
print("RMSE:" , np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))
#model overfit yaptı mı kontrol etmek için
print("Train R2:", model1.score(x_train, y_train)) 
print("Test R2:", model1.score(x_test, y_test))

#SVM
print("\n**************SVM****************")
#ölçeklendirme scaling: modelin adil öğrenmesini sağlamak- featureları eşitlemek amacımız
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_train_scaled = sc1.fit_transform(x_train)
x_test_scaled = sc1.transform(x_test)


sc2 = StandardScaler()
y_train_scaled = sc2.fit_transform(y_train.values.reshape(-1,1)) #-1 satırı otomatik belirle 1 de feature colomnu al 2D array yap 

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

parametreler = {
    "kernel": ["rbf" , "linear", "sigmoid", "poly"],
    "degree" : [1,3,5],
    "gamma":["scale", "auto"]
    }

grid = GridSearchCV(SVR(), parametreler, cv =5, scoring = "r2")
grid.fit(x_train_scaled, y_train_scaled.ravel())
print("En İyi Parametreler:" , grid.best_params_)

model2 = grid.best_estimator_
y_pred_scaled = model2.predict(x_test_scaled)
y_pred = sc2.inverse_transform(y_pred_scaled.reshape(-1,1))

print("\nMSE:" , mean_squared_error(y_test, y_pred))
print("RMSE:" , np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:" , r2_score(y_test, y_pred))


#Random Forest
print("\n**************Random Forest****************")

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

parametreler ={
    "n_estimators": [100,200],
    "max_depth" : [None, 2,5],
    "min_samples_split": [2,5],
    "min_samples_leaf": [1,2],
    "max_features": ["sqrt", "log2"],
    }
grid = GridSearchCV(RandomForestRegressor(), parametreler, cv = 3, scoring="r2")
grid.fit(x_train, y_train)
print("En İyi Parametreler:" , grid.best_params_)

model3 = grid.best_estimator_
y_pred = model3.predict(x_test)

print("\nMSE:" , mean_squared_error(y_test, y_pred))
print("RMSE:" , np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))





