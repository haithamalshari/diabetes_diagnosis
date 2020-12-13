'''
======================================================================================
|                                Diabetes Diagnosis                                  |
|                                   XGBoost Model                                    |
|                                 By Haithm Alshari                                  |
======================================================================================
'''

# Import Libraries
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle as pkl

# Import Dev and Test datasets
dev_df = pd.read_csv('Datasets/Final Dev Data.csv',index_col='SEQN')
test_df = pd.read_csv('Datasets/Final Test Data.csv',index_col='SEQN')

# Drop the borderline entries
dev_df.drop(dev_df[dev_df['Diabetes'] == 2 ].index, inplace = True)
test_df.drop(test_df[test_df['Diabetes'] == 2 ].index, inplace = True)

# Scale the continuous data
col = ['Age','BMI','Workout_mins_in_a_week','Alcohol','Income_Ratio_to_Poverty'] 

def scaling (cols , devdf,testdf):
    from sklearn.preprocessing import StandardScaler
    scalers = []
    for i in cols:
        sc1 = StandardScaler()
        devdf[i] = sc1.fit_transform(np.asarray(devdf[i]).reshape(-1, 1))
        testdf[i] = sc1.transform(np.asarray(testdf[i]).reshape(-1, 1))
        scalers.append([i,sc1])
    return devdf, testdf, scalers

dev_df, test_df, scalers = scaling(col,dev_df,test_df)

# Turn the values in the categorical features into integers
def cat_to_int (cols, devdf, testdf):
    for j in [column for column in dev_df.columns if column not in cols]:
        devdf[j] = devdf[j].astype('int')
        testdf[j] = testdf[j].astype('int')
    return devdf, testdf

dev_df, test_df= cat_to_int(col,dev_df,test_df)

# Split the independent (features) and dependent (lables) variables.
# Then one-hot encode the categorical variable with more than 2 categories
def feat_label_and_one_hot (df):
    cont_var = df.iloc[:,:7].values
    cat_var = df.iloc[:,7:-1].values
    from sklearn.preprocessing import OneHotEncoder
    afi = np.array(range(1,16))
    afi = np.delete(afi, np.s_[10,11,12], axis=0) 
    colls = [afi,np.array(range(0,4)),np.array(range(0,4)),np.array(range(0,5)),np.array(range(1,7)),
             np.array(range(1,6)),np.array(range(0,4)),np.array(range(0,6)),[0,1,2,3,4,5],np.array(range(1,6)),
             np.array(range(1,8)),np.array(range(1,6)),np.array(range(0,4)),[0,1,2,5]]
    onehotencoder = OneHotEncoder(categories = colls)
    cat_var = onehotencoder.fit_transform(cat_var).toarray()
    features = np.concatenate((cont_var,cat_var),axis=1)   
    labels = df.iloc[:,-1].values
    return features, labels

dev_feats, dev_labels = feat_label_and_one_hot (dev_df)
test_feats, test_labels = feat_label_and_one_hot (test_df)

# Splitting the dev set into train, validation
from sklearn.model_selection import train_test_split
train_feats, val_feats, train_labels, val_labels = train_test_split(dev_feats, dev_labels, test_size=0.2, random_state=1)


# Summarize class distribution (Prevalence) in the training set
from collections import Counter
counter = Counter(train_labels)
ratio = counter[0] / counter[1]
print('Diabetes Prevalence in the training data: %.3f' % ratio)

# Define the Evaluation Metrics
def evaluation_metrics(y_val,y_pred):
    from sklearn.metrics import (confusion_matrix, accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score)
    cm = confusion_matrix(y_val, y_pred)
    
    accu = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    f1score = f1_score(y_val, y_pred)
    sensitivity = recall.copy()
    AUC_ROC = roc_auc_score(y_val, y_pred)
    specificity = cm[0,0]/(cm[0,0]+cm[0,1])
    fn = cm[1,0]
    fp = cm[0,1]
    true = cm[0,0] + cm[1,1]
    false = fn  + fp
    return cm,accu,recall,precision,f1score,specificity,sensitivity,AUC_ROC,fn,fp,true,false

# Build the model
classifier = xgb.XGBClassifier(n_estimators = 150, learning_rate = 0.16 , max_depth = 3, min_child_weight = 1, 
                              colsample_bytree = 0.2, subsample = 1, scale_pos_weight =ratio)

# Train the model
classifier.fit(train_feats, train_labels)

# Save the model to file
pkl.dump(classifier, open("xgb_model.pickle.dat", "wb"))

# Load the model from file
classifier = pkl.load(open("xgb_model.pickle.dat", "rb"))

# Predict the validation set results
pred_val = classifier.predict(val_feats)

# Compute the evaluation metrics for the validation dataset
cm,accuracy,recall,precision,f1score,specificity,sensitivity,AUC_ROC,FN,FP,correct,incorrect = evaluation_metrics(val_labels,pred_val)
print('Validation \nAccuracy: {:.4f}, Sensitivity: {:.4f}, Specificity: {:.4f}.'.format(accuracy,sensitivity,specificity))

# Predict the test set results
pred_test = classifier.predict(test_feats)

# Compute the evaluation metrics for the test dataset
cmt,accuracyt,recallt,precisiont,f1scoret,specificityt,sensitivityt,AUC_ROCt,FNt,FPt,correctt,incorrectt = evaluation_metrics(test_labels,pred_test)
print('Test \nAccuracy: {:.4f}, Sensitivity: {:.4f}, Specificity: {:.4f}.'.format(accuracyt,sensitivityt,specificityt))
