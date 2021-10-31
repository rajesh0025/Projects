from scipy.sparse import data
import streamlit as st 
from sklearn import datasets
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA 



st.title("Machine Learning Web Application")

st.write("""
# Explore different classifiers
which one is the best?
""")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris Dataset","Breast Cancer Dataset","Wine Datasets"))

classifier_name =  st.sidebar.selectbox("Select classifier", ("KNN","SVM","RF"))


def get_dataset(dataset_name):
    if dataset_name == "Iris Dataset":
        data = datasets.load_iris()

    elif dataset_name == "Breast Cancer Dataset":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X,y,data

X,y,data= get_dataset(dataset_name)
st.write("Shape of dataset",X.shape)
st.write("No of unique classes",len(np.unique(y)))

st.write("Countplot of classes ..")
sns.countplot(y,label = 'Count')
st.pyplot()

def add_parameters_ui(clf_name):
    params = dict()
    if clf_name =="KNN":
        K = st.sidebar.slider("K",1,15)#Number of neighbors to use 
        params["K"]=K
    elif clf_name =="SVM":
        C = st.sidebar.slider("C",0.01,10.0 )#Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive
        params["C"]=C
    else:
        max_depth= st.sidebar.slider("max_depth",2,15)# the depth of each tree in the forest
        n_estimators = st.sidebar.slider("no of estimators",1,100)# the number of trees to be used in the forest
        params["max_depth"]=max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameters_ui(classifier_name)

def getclassifier(clf_name,params):
    if clf_name =="KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name =="SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"], random_state=12)

        
    return clf

clf = getclassifier(classifier_name,params)

# Classification
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=122)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"classifier={classifier_name}")
st.write(f"accuracy = {acc}")
# st.write(sns.countplot(data.target))

# PLOT
pca = PCA(2)
X_projected = pca.fit_transform(X) 
st.write(X_projected)
x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()
plt.scatter(x1,x2,c=y, alpha =0.8, cmap = "viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

#plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()