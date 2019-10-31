# ML-project-1


here is my code in python 3

import pandas as pd
import matplotlib.pyplot as mat
import numpy as np
import seaborn as sns
import os






print(os.listdir("C:\\Users\\DHANA\\Desktop\\DATA"))#ALL DATA IS IN THIS DIRECTORY
train= pd.read_csv("C:\\Users\\DHANA\\Desktop\\DATA\\X_train.csv.zip" , squeeze =True)#reading and storing data 
test= pd.read_csv("C:\\Users\\DHANA\\Desktop\\DATA\\X_test.csv.zip", squeeze=True)
y_train= pd.read_csv("C:\\Users\\DHANA\\Downloads\\y_train.csv", squeeze= True)

y_train["surface"]=y_train["surface"].map({'fine_concrete':'A','concrete':'B','soft_tiles':'C', 'tiled':'D','soft_pvc':'E','hard_tiles_large_space':'F', 'carpet':'G', 'hard_tiles':'H', 'wood':'I'})
train= pd.merge(train,y_train[['surface','series_id']],on="series_id")
m_data=pd.read_csv("C:\\Users\\DHANA\\Documents\\m_data.csv",squeeze=True)
print(train.head())#for testing weter added or not

"""data exploration: this is a multiclass classification with imbalanced dataset
let us see how imbalanced the data is..!!"""

print(y_train["surface"].value_counts())
#data to plot
mat.figure(figsize = (12, 8))
sns.countplot(y_train["surface"])
mat.title("Number of datapoints per class")
mat.ylabel("Number of datapoints")
mat.xlabel("surface name")
mat.show()

#traindata

train_data= train.drop(['surface'], axis=1)
print(train_data.head())



#usage of feature engineering
def feat_eng(data):
    df=pd.DataFrame()
    data['totl_anglr_vel']=(data['angular_velocity_X']**2 + data['angular_velocity_Y']**2+ data['angular_velocity_Z'])**0.5
    data['totl_linr_acc']=(data['linear_acceleration_X']**2+ data['linear_acceleration_Y']**2+data['linear_acceleration_Z'])**0.5
    data['totl_xyz'] = (data['orientation_X']**2 + data['orientation_Y']**2 + data['orientation_Z'])**0.5
    data['acc_vs_vel'] = data['totl_linr_acc'] / data['totl_anglr_vel']
    for col in data.columns:
        if col in ['row_id','series_id','measurement_number']:
            continue
        df[col + '_median'] = data.groupby(['series_id'])[col].median()
        df[col + '_max'] = data.groupby(['series_id'])[col].max()
        
    return df


 
X_train = feat_eng(train_data)
test_data = feat_eng(test)
print(X_train.head())

#plotting of confusion matrix

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

class_names=y_train.surface
G=X_train.fillna(X_train.mean())
D=y_train.surface

from sklearn.model_selection import train_test_split

#split the data into train and test
X_train, X_test, y_train,y_test=train_test_split( G, D, random_state=0)
#run classifier using a model that is regularised
#so train a logistic regressionmodel on the training set
from sklearn.linear_model import LogisticRegression 
logreg=LogisticRegression()
#fit model
logreg.fit(X_train, y_train)
#predict test data
y_pred=logreg.predict(X_test)




results=confusion_matrix(y_test,y_pred)
print(results)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
""" filter warnings"""


#undersampling 

from imblearn.under_sampling import ClusterCentroids
cc=ClusterCentroids(random_state=0)
G_resampled,D_resampled = cc.fit_resample(G,D)


#split the data into train and test
X_train, X_test, y_train,y_test=train_test_split( G_resampled, D_resampled, random_state=0)
#run classifier using a model that is regularised
#so train a logistic regressionmodel on the training set
from sklearn.linear_model import LogisticRegression 
logreg=LogisticRegression()
#fit model
logreg.fit(X_train, y_train)
#predict test data
y_pred=logreg.predict(X_test)
from sklearn import tree
clas=tree.DecisionTreeClassifier()
clas.fit(X_train, y_train)
y_pred=clas.predict(X_test)


results=confusion_matrix(y_test,y_pred)
print(results)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
""" filter warnings"""  

# randomForestclasifier


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(random_state=0)
rfc.fit(G_resampled, D_resampled)
RandomForestClassifier(bootstrap=True, random_state=0)
y_pred=rfc.predict(X_test)

results=confusion_matrix(y_test,y_pred)
print(results)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))



