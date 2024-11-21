import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import resample
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, ConfusionMatrixDisplay, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
import shap
from sklearn.ensemble import RandomForestClassifier
import time
from tabulate import tabulate
from lime.lime_tabular import LimeTabularExplainer

#this code follow notebook,https://www.kaggle.com/code/chandrimad31/rainfall-prediction-7-popular-models, instruction

SanDiegoPath = "San Diego.csv"
NYPath = "NY.csv"
#"/Users/ringuyen/Desktop/rain forecast/dataset/NY.csv"

SanDiegoData = pd.read_csv(SanDiegoPath)
NYData = pd.read_csv(NYPath)
#fig = plt.figure(figsize = (8,5))




#handle imbalance by oversample positive case
def classimbalance(Data, Path):
    # plot to see original data have class imbalance
    Data['Rain Tomorrow'].value_counts(normalize = True).plot(kind='bar', color= ['skyblue','navy'], alpha = 0.9, rot=0)
    plt.title('RainTomorrow Indicator No(0) and Yes(1) in ' + Path.replace(".csv",""))
    plt.show()

    no = Data[Data['Rain Tomorrow'] == 0]
    yes = Data[Data['Rain Tomorrow'] == 1]
    yesOverSampled = resample(yes, replace=True, n_samples = len(no), random_state=123)
    overSample = pd.concat([no,yesOverSampled])

    #plot to see oversample solve imbalance#
    overSample['Rain Tomorrow'].value_counts(normalize = True).plot(kind='bar', color= ['skyblue','navy'], alpha = 0.9, rot=0)
    plt.title('RainTomorrow Indicator No(0) and Yes(1) in ' + Path.replace(".csv",""))
    plt.show()
    return overSample

overSampleSanDiego = classimbalance(SanDiegoData, SanDiegoPath)
overSampleNY = classimbalance(NYData, NYPath)

# There is no missing data so no need to impute
# overSample.select_dtypes(include=['object']).columns
# overSample['Date'] = overSample['Date'].fillna(overSample['Date'].mode()[0])
# overSample['Location'] = overSample['Location'].fillna(overSample['Location'].mode()[0])

#label encoding as data preprocessing
def labelEncoding(data):
    lencoders = {}
    for col in data.select_dtypes(include=['object']).columns:
        lencoders[col] = LabelEncoder()
        data[col] = lencoders[col].fit_transform(data[col])

labelEncoding(overSampleNY)
labelEncoding(overSampleSanDiego)

def removeOutlier(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    print(data.shape)
    # remove outlier base on IQR
    data = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
    print(data.shape)
removeOutlier(overSampleNY)

# heatmap for corelation
def heatMapCorr(data):
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    f, ax = plt.subplots(figsize=(20, 20))
    cmap = sns.diverging_palette(250, 25, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=None, center=0,square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .9})
    plt.show()
heatMapCorr(overSampleNY)

features = overSampleNY[['Date','Location', 'Temperature','Humidity','Wind Speed','Precipitation','Cloud Cover','Pressure']]
target = overSampleNY['Rain Tomorrow']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=1)
#Normalize by MinMaxScaler
X_train = MinMaxScaler().fit_transform(X_train)
X_test = MinMaxScaler().fit_transform(X_test)

def plotRoc(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def runModel(model, X_train, y_train, X_test, y_test, verbose=True):
    if verbose == False:
        model.fit(X_train,y_train, verbose=0)
    else:
        model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred) 
    
    print("Accuracy = {}".format(accuracy))
    print("ROC Area under Curve = {}".format(roc_auc))
    print(classification_report(y_test,y_pred,digits=5))
    
    probs = model.predict_proba(X_test)  
    probs = probs[:, 1]  
    fper, tper, thresholds = roc_curve(y_test, probs) 
    plotRoc(fper, tper)

    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,cmap=plt.cm.Blues, normalize = 'all')
    plt.show()
    return model, accuracy, roc_auc

params_lr = {'penalty': 'l1', 'solver':'liblinear'}

model_lr = LogisticRegression(**params_lr)
model_lr, accuracy_lr, roc_auc_lr = runModel(model_lr, X_train, y_train, X_test, y_test)

params_rf = {'max_depth': 16,
             'min_samples_leaf': 1,
             'min_samples_split': 2,
             'n_estimators': 100,
             'random_state': 12345}

model_rf = RandomForestClassifier(**params_rf)
model_rf, accuracy_rf, roc_auc_rf = runModel(model_rf, X_train, y_train, X_test, y_test)

params_xgb ={'n_estimators': 500,
            'max_depth': 16}

model_xgb = xgb.XGBClassifier(**params_xgb)
model_xgb, accuracy_xgb, roc_auc_xgb = runModel(model_xgb,X_train,y_train,X_test,y_test)

accuracy_scores = [accuracy_lr, accuracy_rf, accuracy_xgb]
roc_auc_scores = [roc_auc_lr, roc_auc_rf, roc_auc_xgb]

dataFrame = {'Model':['Logistic Regression', 'Random Forest', 'XGBoost']
                    ,'Accuracy':accuracy_scores
                    ,'Roc_Auc':roc_auc_scores}
dataFrame = pd.DataFrame(dataFrame)

sns.barplot(dataFrame, x='Model', y='Accuracy',palette='summer')
plt.show()
sns.barplot(dataFrame, x='Model', y='Roc_Auc',palette='summer')
plt.show()
# explainer = shap.TreeExplainer(model_xgb, X_test)
# shap_values = explainer(X_test)

# shap.summary_plot(shap_values,features, plot_type="bar")

# class_names = ['rain tomorrow', 'No rain tomorrow']
# feature_names = ['Date','Location', 'Temperature','Humidity','Wind Speed','Precipitation','Cloud Cover','Pressure']
# explainer = LimeTabularExplainer(X_train, feature_names =     
#                                  feature_names,
#                                  class_names = class_names, 
#                                  mode = 'classification')
# for i in range(20):
#     explaination = explainer.explain_instance(
#         data_row=X_test[i],
#         predict_fn=model_xgb.predict_proba,
#         num_features=30
#     )
# fig = explaination.as_pyplot_figure()
# plt.tight_layout()
# plt.show()
