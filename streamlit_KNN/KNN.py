import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score,auc,roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
st.set_option("deprecation.showPyplotGlobalUse",False)

st.title("KNN with hyper parameter tuning and Evaluation")

# upload file
file_upload = st.sidebar.file_uploader("Upload Only preprosesing data")

if file_upload is not None:
    data=pd.read_csv(file_upload)

    #Traget column
    target_columns = st.selectbox("Slect your target columns",options=data.columns)

    # input and output columns
    X=data.drop(columns=[target_columns])
    y=data[target_columns]

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)

    # logistic Regression hyperparameters
    st.sidebar.header("Hyperparameter of KNN ")
    leaf_size = st.sidebar.slider(" Leaf Size", min_value=5,max_value=100,step=5,value=40)
    n_neighbours = st.sidebar.slider("number of Neighburs",min_value=1,max_value=10,step=1,value=3)
    p = st.sidebar.slider("p",min_value=1,max_value=5,step=1,value=2)
    matrix=st.sidebar.selectbox("Distance matrix",options=["minkowski", "manhattan","euclidean","chebyshev","cosine"],index=0)
    

    # load the model
    KNN = KNeighborsClassifier(p=p,n_neighbors=n_neighbours,leaf_size=leaf_size,metric=matrix,n_jobs=-1)
    KNN.fit(X_train,y_train)

    # predict the model
    y_pred=KNN.predict(X_test)

    # calculate accuracy,precision,F1_score,recall
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    recall= recall_score(y_test,y_pred)

    st.write("Accuracy Score:",accuracy)
    st.write("Precision Score:",precision)
    st.write("Recall Score:",recall)
    st.write("F1 Score:",f1)


    # plot confusion matrix
    cm = confusion_matrix(y_test,y_pred)
    display = ConfusionMatrixDisplay(cm,display_labels=[False,True])
    display.plot()
    plt.grid(False)
    plt.show()
    st.pyplot()


    # plot auc- roc curve
    
    y_pred_prob=KNN.predict_proba(X_test)[:,1] # class-1 probablites
    fpr, tpr,thhershold=roc_curve(y_test,y_pred_prob)
    plt.plot([0,1],[0,1], color="red",lw=2,label="Average model")
    plt.plot(fpr,tpr,color="green",lw=2,label="KNN_model_with_hyperparameter")
    plt.xlabel("Flase_positive_Rate")
    plt.ylabel("True_positive_Rate")
    plt.title("roc_auc_curve")
    plt.legend()
    plt.show()
    st.pyplot()


    # Area under the curve

    st.write("Computed area under the curve (AUC)",(auc(fpr,tpr)))



