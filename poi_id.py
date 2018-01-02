#!/usr/bin/python

import sys
import pickle
import pandas as pd
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from sklearn.preprocessing import MinMaxScaler
from tester import dump_classifier_and_data

### Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                    # 'salary',
                    # 'to_messages',
                    # 'deferral_payments',
                    # 'total_payments',
                    # 'loan_advances',
                    # 'bonus', #massive improvement when this feature is removed! May be impacted by outliers?
                    # 'email_address',
                    # 'restricted_stock_deferred',
                    # 'deferred_income',
                    # 'total_stock_value',
                    'expenses',
                    # 'from_poi_to_this_person',
                    'exercised_stock_options',
                    # 'from_messages',
                    'other',
                    # 'from_this_person_to_poi',
                    # 'long_term_incentive',
                    # 'shared_receipt_with_poi',
                    # 'restricted_stock',
                    # 'director_fees',
                    # 'payments_salary_ratio',
                    'from_ratio',
                    # 'to_ratio'
                    ]


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

'''
DATA EXPLORATION & MUNGING
'''
# print(data_dict)
### Explore the data using pandas

# Transform into a dataframe using pandas
df = pd.DataFrame.from_dict(data_dict, orient= 'index')

# Change datatypes from 'objects' to 'floats' so that they are easier to handle with pandas
df = df.drop(['email_address'], axis= 1).astype(float)

# Print some summary statistics for all the variables
print(df.describe())

# Count the number of missing values per variable
for col in df.columns:
    print(col, df[col].isnull().sum())

# Identify outliers
print(df.loc[:,'salary'].max())
print(df.loc[:,'salary'].argmax()) # 'TOTAL' value has snuck in as highest salary - need to remove

### Remove outliers
data_dict.pop("TOTAL")

'''
FEATURE ENGINEERING
'''
### Create new feature(s)
def add_new_features(data):

    '''
    Takes in a dictionary of employees (keys), where the value of each observation ('row') is itself a dictionary
    Calculates 3 new features ['payments_salary_ratio', 'from_ratio', 'to_ratio'] to possibly improve predictions
    Returns the data dictionary with the additional calculated features
    '''

    for row in data.values():

        try:
            row['payments_salary_ratio'] = row['total_payments'] / row['salary']
        except:
            row['payments_salary_ratio'] = 'NaN'

        try:
            row['from_ratio'] = row['from_this_person_to_poi'] / row['from_messages']
        except:
            row['from_ratio'] = 'NaN'

        try:
            row['to_ratio'] = row['from_poi_to_this_person'] / row['to_messages']
        except:
            row['to_ratio'] = 'NaN'

    return data

# Call the add_new_features() func on the data_dict and store it as a my_dataset var
my_dataset = add_new_features(data_dict)
# print(my_dataset)
# print(len(my_dataset))

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Scale my features - useful for some ML models, such as KNN
scaler = MinMaxScaler()
features = scaler.fit_transform(features)


'''
MODEL BUILDING AND EVALUTION
'''
### Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# clf = GaussianNB()
# clf = DecisionTreeClassifier(min_samples_split= 40)
# clf = RandomForestClassifier(max_depth= 5, random_state= 0)
clf = AdaBoostClassifier(n_estimators= 8, learning_rate = 0.7, random_state= 1)
# clf = GradientBoostingClassifier(n_estimators= 20, learning_rate = 1.0, max_depth= 1, random_state= 0)
# clf = KNeighborsClassifier(n_neighbors= 5)

### Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

'''
Best model tested:
clf = AdaBoostClassifier(n_estimators= 8, learning_rate = 0.7, random_state= 1)
features = ['expenses', 'exercised_stock_options', 'other', 'from_ratio']
P: 0.63 | R: 0.45
'''

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

### Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
