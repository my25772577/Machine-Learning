# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:00:42 2019

@author: YM60580
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
#先把这个数据清理下，把复杂的education精简
dd = pd.read_csv('banking.csv')
dd = dd.dropna()
dd['education'] = np.where(dd['education'] == 'basic.9y', 'Basic', dd['education'])
dd['education'] = np.where(dd['education'] == 'basic.6y', 'Basic', dd['education'])
dd['education'] = np.where(dd['education'] == 'basic.4y', 'Basic', dd['education'])

#看看y的值是多少， 这里y已经在data集里定义了y
dd['y'].value_counts()

#看看图
import seaborn as sns
sns.countplot(x = 'y', data = dd, palette = 'hls')

#计算数据集中sub这个服务的比例
count_no_sub = len(dd[dd['y']==0])
count_sub = len(dd[dd['y']==1])
pct_of_no_sub = count_no_sub/(count_no_sub + count_sub)
print("percentage of no subscription is", pct_of_no_sub*100)

pct_of_sub = count_sub/(count_no_sub + count_sub)
print("percentage of subscription is", pct_of_sub*100)

#以上数据显示类比上很不平衡，所以需要进一步看看data的情况
groupbyby = dd.groupby('y').mean()
groupbyjob = dd.groupby('job').mean()
groupbymarry =  dd.groupby('marital').mean()
groupbyedu = dd.groupby('education').mean()

#visualization the job vs y
%matplotlib inline
pd.crosstab(dd.job, dd.y).plot(kind = 'bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')

#visualization the maritals vs y
table = pd.crosstab(dd.marital, dd.y)
table.div(table.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True)
plt.title('Purchase Ratio for Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Ratio of Purchase')

#看education的情况
table1 = pd.crosstab(dd.education, dd.y)
table1.div(table1.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Purchase Ratio for Education Status')
plt.xlabel('Education Status')
plt.ylabel('Ratio of Purchase')

#看看days of the week
table2 = pd.crosstab(dd.day_of_week, dd.y)
table2.div(table2.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True)
plt.title('Purchase Ratio for day_of_week Status')
plt.xlabel('day_of_week')
plt.ylabel('Ratio of Purchase')

#看看month
pd.crosstab(dd.month,dd.y).plot(kind = 'bar')
plt.title('Purchase Ratio for Month Status')
plt.xlabel('Month')
plt.ylabel('Ratio of Purchase')

#看看age
dd.age.hist()
plt.title('Purchase for Age')
plt.xlabel('Age')
plt.ylabel('Frequency of Purchase')

#看看前一次推销的的结果
pd.crosstab(dd.poutcome, dd.y).plot(kind = 'bar')
plt.title('Purchase Ratio for Poutcome Status')
plt.xlabel('Poutcome')
plt.ylabel('Ratio of Purchase')

#create dummy variabels 
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var  
    cat_list = pd.get_dummies(dd[var], prefix=var)
    data1=dd.join(cat_list)
    dd=data1


cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=dd.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
data_final=dd[to_keep]
data_final.columns.values

#split the data
X = data_final.loc[:, data_final.columns != 'y']
y = data_final.loc[:, data_final.columns == 'y']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# import the SMOTE over sampling method
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state =0)

columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])


# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))

#用RFE方法找出每个variable的排序
data_final_vars = data_final.columns.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
rfe = RFE(classifier, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

#把相关性高的varible取出来
cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 'default_no', 'default_unknown', 
      'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 
      'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"] 
X=os_data_X[cols]
y=os_data_y['y']

#implement the model
import statsmodels.api as sm
logis_model = sm.Logit(y, X)
y_pre = logis_model.fit()
print(y_pre.summary2())

#move掉P value 大于0.05的
cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 
      'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 
      'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"] 
X=os_data_X[cols]
y=os_data_y['y']
logis_model = sm.Logit(y, X)
result = logis_model.fit()
print(result.summary2())

#最后运用ML进行predict·
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =0)
classifier.fit(X_train, y_train)


#prediction and making confusion matrix
y_pred = classifier.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


#看一下report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#看ROC曲线
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

















