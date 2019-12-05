# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:21:52 2019

@author: YM60580
"""

#题目要求
#1.Predict conversion rate 
#2.Come up with recommendations to improve conversion rate
#拿到题目先看，这道题是回归问题还是分类问题，界定的区别是y是1/0 还是一个具体的数值，这里是分类问题，所以我们可以用很多模型

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
 
#第一步，读取数据，看看有没有要清理的部分
dd = pd.read_csv('conversion_data.csv')
#有5个variable， 1个independent variable, 看起来所有的variable目前都不需要去掉，我们先去掉一些na的data
dd = dd.dropna()
#这一步很重要，把na清理掉，但是看起来这个数据集没有na
print(dd.shape)
dd.head()
dd['country'].unique()
dd['source'].unique()

# 5个variable中， country， source 还有 new user 是category varibale,age 和total page点击数是numeric
# 看看y的值， conversion rate
dd['converted'].value_counts()
10200/306000
#目前的conversion rate是3.33%
#画个图看看， seaborn很有用
import seaborn as sns 
sns.countplot(x = 'converted', data = dd, palette = 'hls')
#看conversion rate， 需要精确计算， 以上的只是约值，以下更精确
no_cov = len(dd[dd['converted']==0])
cov = len(dd[dd['converted']==1])
pct_of_cov = cov/(no_cov+cov)
print("conversion rate is:", pct_of_cov * 100)

#进一步看看data,注意，这里他只计算numberic的组， 所以country和source并没有被计算
dd.groupby('converted').mean()

#所以再下一步就是看看category variable的分布
dd.groupby('country').mean()
dd.groupby('source').mean()
%matplotlib inline
# pd.crosstab(dd.country, dd.converted).plot(kind = 'bar',stacked = True) 这个代码适合category比较多的去比较， 以下的吧每个分类总数变成1，然后计算每个里的比例，更清楚易懂
table = pd.crosstab(dd.country,dd.converted)
table.div(table.sum(1).astype(float), axis =0).plot(kind = 'bar', stacked = True)
#China的转换率特别低，所以这个variable相关
table1 = pd.crosstab(dd.source, dd.converted)
table1.div(table1.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True)
#感觉差别不是很大， 但是因为这题的varibale 比较少，所以先放着
table2 = pd.crosstab(dd.new_user, dd.converted)
table2.div(table2.sum(1).astype(float), axis =0).plot(kind = 'bar', stacked = True)
#以上可见，new_user也影响很大
dd.age.hist()
dd.total_pages_visited.hist()

#对dummy variable进行encoding
cat_var = ['country', 'source']
for var in cat_var:
    cat_list = 'var' + '_' + var
    cat_list = pd.get_dummies(dd[var], prefix = var)
    dd1 = dd.join(cat_list)
    dd = dd1

cat_var = ['country', 'source'] 
data_vars=dd.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_var]
data_final=dd[to_keep]
data_final.columns.values

#split the data
from sklearn.model_selection import train_test_split
X = data_final.iloc[:, data_final.columns != 'converted']
y = data_final.iloc[:, data_final.columns == 'converted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#oversampling 因为converted的数据太少了，所以要人为的增加样本范围
from imblearn.over_sampling import SMOTE
oversample = SMOTE(random_state = 0)

columns = X_train.columns
os_data_X, os_data_y = oversample.fit_sample(X_train,y_train)
os_data_X = pd.DataFrame(data = os_data_X, columns = columns)
os_data_y = pd.DataFrame(data = os_data_y, columns = ['converted'])

# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['converted']==0]))
print("Number of subscription",len(os_data_y[os_data_y['converted']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['converted']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['converted']==1])/len(os_data_X))

#用RFE 找出每个variable的影响性排序
data_final_vars = data_final.columns.tolist()
y=['converted']
X=[i for i in data_final_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
rfe = RFE(classifier, 8)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

#看看p-value， 发现germany影响不大
cols=['new_user','country_China', 'country_Germany','country_UK','country_US', 'source_Ads','source_Direct','source_Seo'] 
X=os_data_X[cols]
y=os_data_y['converted']
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

#去掉Germany
cols=['new_user','country_China','country_UK','country_US', 'source_Ads','source_Direct','source_Seo'] 
X=os_data_X[cols]
y=os_data_y['converted']
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

#fit the model, choose logisticregression here
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
X_train,X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.3, random_state = 0)
classifier.fit(X_train, y_train)

#preidct y_pred
y_pred = classifier.predict(X_test)
accuracy = classifier.score(X_test, y_test)
print('Accuracy of logistic regression lassifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

#make confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#看下report
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