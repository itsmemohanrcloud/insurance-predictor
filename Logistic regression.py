#Libraries used

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pylab import rcParams

#Read Dataset

data = pd.read_csv('CustomerChurnTestDataset.csv')
#df_data.head(10)

#Data Plot

customerChurn=data.groupby('Age').count()
plt.bar(customerChurn.index.values, customerChurn['Left'])
plt.xlabel('Age')
plt.ylabel('Left')
plt.show()
data.Left.value_counts()

customerChurn=data.groupby('Referred Friends').count()
plt.bar(customerChurn.index.values, customerChurn['Left'])
plt.xlabel('Referred Friends')
plt.ylabel('Left')
plt.show()

customerChurn=data.groupby('Term Duration').count()
plt.bar(customerChurn.index.values, customerChurn['Left'])
plt.xlabel('Term Duration')
plt.ylabel('Left')
plt.show()

customerChurn=data.groupby('Critical Customer Complaints').count()
plt.bar(customerChurn.index.values, customerChurn['Left'])
plt.xlabel('Critical Customer Complaints')
plt.ylabel('Left')
plt.show()

customerChurn=data.groupby('Gender').count()
plt.bar(customerChurn.index.values, customerChurn['Left'])
plt.xlabel('Gender')
plt.ylabel('Left')
plt.show()


customerChurn=data.groupby('Martial Status').count()
plt.bar(customerChurn.index.values, customerChurn['Left'])
plt.xlabel('Martial Status')
plt.ylabel('Left')
plt.show()


data.info()

#Finding How much customer Left
data.Left.value_counts()


#Converting categorical data into numeric ones
data['Insurance Plan Type'].replace(['Individual','Family'],[0,1],inplace=True)
data['Martial Status'].replace(['Unmarried','Married'],[1,0],inplace=True)
data['Gender'].replace(['Male','Female'],[1,0],inplace=True)
data['competitor Equivalent offer'].replace(['yes','No'],[1,0],inplace=True)
data['Referred Friends'].replace(['yes','No'],[1,0],inplace=True)
data['Critical Customer Complaints'].replace(['yes','No'],[1,0],inplace=True)
data['Customer Satisfaction'].replace(['Satisfied','Dissatisfied'],[1,0],inplace=True)



data['competitor Equivalent offer'] = pd.to_numeric(data['Referred Friends'], errors = 'coerce')
data.loc[data['competitor Equivalent offer'].isna()==True]
data[data['competitor Equivalent offer'].isna()==True] = 0

#heatMap Depiction
corr = data.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12})
heat_map=plt.gcf()
heat_map.set_size_inches(20,15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()



from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.30)

train_y = train['Left']
test_y = test['Left']

train_x = train
train_x.pop('Left')
test_x = test
test_x.pop('Left')

#Start Logistic Regression

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(X=train_x, y=train_y)
 
test_y_pred = logisticRegr.predict(test_x)
confusion_matrix = confusion_matrix(test_y, test_y_pred)
print('Intercept: ' + str(logisticRegr.intercept_))
print('Regression: ' + str(logisticRegr.coef_))
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logisticRegr.score(test_x, test_y)))
print(classification_report(test_y, test_y_pred))
 
confusion_matrix_df = pd.DataFrame(confusion_matrix, ('Not Left', 'Left'), ('Not Left', 'Left'))
heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={"size": 20}, fmt="d")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize = 14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize = 14)
plt.ylabel('Predicted label', fontsize = 14)
plt.xlabel('True label', fontsize = 14)

# churn count

data['Left'].value_counts()



#balancing churn dataset using upsampling
from sklearn.utils import resample

data_majority = data[data['Left']==0]
data_minority = data[data['Left']==1]

data_minority_upsampled = resample(data_minority,replace=True,
n_samples=13, #same number of samples as majority classe
random_state=1) #set the seed for random resampling
# Combine resampled results
data_upsampled = pd.concat([data_majority, data_minority_upsampled])

data_upsampled['Left'].value_counts()

 #working on balanced dataset after upsampling

train, test = train_test_split(data_upsampled, test_size = 0.25)
 
train_y_upsampled = train['Left']
test_y_upsampled = test['Left']
 
train_x_upsampled = train
train_x_upsampled.pop('Left')
test_x_upsampled = test
test_x_upsampled.pop('Left')
 
logisticRegr_balanced = LogisticRegression()
logisticRegr_balanced.fit(X=train_x_upsampled, y=train_y_upsampled)
 
test_y_pred_balanced = logisticRegr_balanced.predict(test_x_upsampled)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logisticRegr_balanced.score(test_x_upsampled, test_y_upsampled)))
print(classification_report(test_y_upsampled, test_y_pred_balanced))


from sklearn.metrics import roc_auc_score

# Get class probabilities for both models
test_y_prob = logisticRegr.predict_proba(test_x)
test_y_prob_balanced = logisticRegr_balanced.predict_proba(test_x_upsampled)

# We only need the probabilities for the positive class
test_y_prob = [p[1] for p in test_y_prob]
test_y_prob_balanced = [p[1] for p in test_y_prob_balanced]

print('Unbalanced model AUROC: ' + str(roc_auc_score(test_y, test_y_prob)))
print('Balanced model AUROC: ' + str(roc_auc_score(test_y_upsampled, test_y_prob_balanced)))

from sklearn.metrics import roc_auc_score

 
# Get class probabilities for both models
test_y_prob = logisticRegr.predict_proba(test_x)
test_y_prob_balanced = logisticRegr_balanced.predict_proba(test_x_upsampled)
 
# We only need the probabilities for the positive class
test_y_prob = [p[1] for p in test_y_prob]
test_y_prob_balanced = [p[1] for p in test_y_prob_balanced]
 
print('Unbalanced model AUROC: ' + str(roc_auc_score(test_y, test_y_prob)))
print('Balanced model AUROC: ' + str(roc_auc_score(test_y_upsampled, test_y_prob_balanced)))

# end of logistic regression


#Decision tree

from sklearn import tree
import graphviz 
 
# Create each decision tree (pruned and unpruned)
decisionTree_unpruned = tree.DecisionTreeClassifier()
decisionTree = tree.DecisionTreeClassifier(max_depth = 4)
 
# Fit each tree to our training data
decisionTree_unpruned = decisionTree_unpruned.fit(X=train_x, y=train_y)
decisionTree = decisionTree.fit(X=train_x, y=train_y)
 
# Generate PDF visual of decision tree
churnTree = tree.export_graphviz(decisionTree, out_file=None, 
                         feature_names = list(train_x.columns.values),  
                         class_names = ['No Left', 'Left'],
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(churnTree)
graph.render('decision_tree.gv', view=True)

test_y_pred_dt = decisionTree.predict(test_x)
print('Accuracy of decision tree classifier on test set: {:.2f}'.format(decisionTree.score(test_x, test_y)))