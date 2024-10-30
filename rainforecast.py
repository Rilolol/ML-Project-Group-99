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

Path = "LA.csv"
#"/Users/ringuyen/Desktop/rain forecast/dataset/NY.csv"

Data = pd.read_csv(Path)
#fig = plt.figure(figsize = (8,5))


# plot to see original data have class imbalance
# NYDataTest['Rain Tomorrow'].value_counts(normalize = True).plot(kind='bar', color= ['skyblue','navy'], alpha = 0.9, rot=0)
# plt.title('RainTomorrow Indicator No(0) and Yes(1) in the Imbalanced Dataset')
# plt.show()

#handle imbalance by oversample positive case
no = Data[Data['Rain Tomorrow'] == 0]
yes = Data[Data['Rain Tomorrow'] == 1]
yesOverSampled = resample(yes, replace=True, n_samples = len(no), random_state=123)
overSample = pd.concat([no,yesOverSampled])

# plot to see oversample solve imbalance
# overSample['Rain Tomorrow'].value_counts(normalize = True).plot(kind='bar', color= ['skyblue','navy'], alpha = 0.9, rot=0)
# plt.title('RainTomorrow Indicator No(0) and Yes(1) in the Imbalanced Dataset')
# plt.show()

# There is no missing data so no need to impute
# overSample.select_dtypes(include=['object']).columns
# overSample['Date'] = overSample['Date'].fillna(overSample['Date'].mode()[0])
# overSample['Location'] = overSample['Location'].fillna(overSample['Location'].mode()[0])

#label encoding as data preprocessing
lencoders = {}
for col in overSample.select_dtypes(include=['object']).columns:
    lencoders[col] = LabelEncoder()
    overSample[col] = lencoders[col].fit_transform(overSample[col])

Q1 = overSample.quantile(0.25)
Q3 = overSample.quantile(0.75)
IQR = Q3 - Q1

# remove outlier base on IQR
overSample = overSample[~((overSample < (Q1 - 1.5 * IQR)) |(overSample > (Q3 + 1.5 * IQR))).any(axis=1)]

# heatmap for corelation
# corr = overSample.corr()
# mask = np.triu(np.ones_like(corr, dtype=np.bool))
# f, ax = plt.subplots(figsize=(20, 20))
# cmap = sns.diverging_palette(250, 25, as_cmap=True)
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=None, center=0,square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .9})
# plt.show()

features = overSample[['Date','Location', 'Temperature','Humidity','Wind Speed','Precipitation','Cloud Cover','Pressure']]
target = overSample['Rain Tomorrow']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=1)
# Normalize by MinMaxScaler
# X_train = MinMaxScaler().fit_transform(X_train)
# X_test = MinMaxScaler().fit_transform(X_test)

def plot_roc_cur(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

def run_model(model, X_train, y_train, X_test, y_test, verbose=True):
    t0=time.time()
    if verbose == False:
        model.fit(X_train,y_train, verbose=0)
    else:
        model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred) 
    coh_kap = cohen_kappa_score(y_test, y_pred)
    time_taken = time.time()-t0
    print("Accuracy = {}".format(accuracy))
    print("ROC Area under Curve = {}".format(roc_auc))
    print("Cohen's Kappa = {}".format(coh_kap))
    print("Time taken = {}".format(time_taken))
    print(classification_report(y_test,y_pred,digits=5))
    
    probs = model.predict_proba(X_test)  
    probs = probs[:, 1]  
    fper, tper, thresholds = roc_curve(y_test, probs) 
    plot_roc_cur(fper, tper)

    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,cmap=plt.cm.Blues, normalize = 'all')
    plt.show()
    return model, accuracy, roc_auc, coh_kap, time_taken

#params_lr = {'penalty': 'l1', 'solver':'liblinear'}

# model_lr = LogisticRegression(**params_lr)
# model_lr, accuracy_lr, roc_auc_lr, coh_kap_lr, tt_lr = run_model(model_lr, X_train, y_train, X_test, y_test)
# shap.initjs()

# params_rf = {'max_depth': 16,
#              'min_samples_leaf': 1,
#              'min_samples_split': 2,
#              'n_estimators': 100,
#              'random_state': 12345}

params_xgb ={'n_estimators': 500,
            'max_depth': 16}

model_xgb = xgb.XGBClassifier(**params_xgb)
model_xgb, accuracy_xgb, roc_auc_xgb, coh_kap_xgb, tt_xgb = run_model(model_xgb, X_train, y_train, X_test, y_test)

explainer = shap.TreeExplainer(model_xgb, X_test)
shap_values = explainer(X_test)

shap.plots.bar(shap_values)

