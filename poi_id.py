#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus',
                 'bonus_salary_ratio',
                 'total_stock_value',
                 'exercised_stock_options',
                 'from_this_person_to_poi_percentage',
                 'shared_receipt_with_poi'] # You will need to use more features
print 'Number of features: ', len(features_list)
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Explore the dataset
total_number_people = len(data_dict)
print 'Number of people in the dataset:', total_number_people

### Number of poi in the dataset
num_poi = 0
for i in data_dict:
    if data_dict[i]['poi']==True:
        num_poi=num_poi+1
print 'Number of poi in the dataset:', num_poi
print 'Number of person who is not poi:', len(data_dict)-num_poi

### Task 2: Remove outliers
features =['salary', 'bonus']
data = featureFormat(data_dict, features)
for point in data:
    salary=point[0]
    bonus=point[1]
    plt.scatter(salary, bonus)
plt.xlabel('salary')
plt.ylabel('bonus')

plt.show() 

for i, v in data_dict.items():
    if v['salary'] != 'NaN' and v['salary'] > 10000000:
        print i
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0 )
data_dict.pop('LOCKHART EUGENE E', 0)
print "Number of datapoint after removing outliers is :", len(data_dict)

features =['salary', 'bonus']
data = featureFormat(data_dict, features)
for point in data:
    salary=point[0]
    bonus=point[1]
    plt.scatter(salary, bonus)
plt.xlabel('salary')
plt.ylabel('bonus')

plt.show()

### Task 3: Create new feature(s)
# Bonus-salary ratio
for employee, features in data_dict.iteritems():
    if features['bonus'] == "NaN" or features['salary'] == "NaN":
        features['bonus_salary_ratio'] = "NaN"
    else:
        features['bonus_salary_ratio'] = float(features['bonus']) / float(features['salary'])

# from_this_person_to_poi as a percentage of from_messages
for employee, features in data_dict.iteritems():
    if features['from_this_person_to_poi'] == "NaN" or features['from_messages'] == "NaN":
        features['from_this_person_to_poi_percentage'] = "NaN"
    else:
        features['from_this_person_to_poi_percentage'] = float(features['from_this_person_to_poi']) / float(features['from_messages'])

# from_poi_to_this_person as a percentage of to_messages
for employee, features in data_dict.iteritems():
    if features['from_poi_to_this_person'] == "NaN" or features['to_messages'] == "NaN":
        features['from_poi_to_this_person_percentage'] = "NaN"
    else:
        features['from_poi_to_this_person_percentage'] = float(features['from_poi_to_this_person']) / float(features['to_messages'])

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

scaler = MinMaxScaler()
select = SelectKBest()
dtc = DecisionTreeClassifier()
svc = SVC()
knc = KNeighborsClassifier()

steps = [('feature_selection', select),('dtc', dtc)]

# Create pipeline
pipeline = Pipeline(steps)

# Parameters to try in grid search
parameters = dict(
                  feature_selection__k=[2, 3, 5, 6], 
                  dtc__criterion=['gini', 'entropy'],
                  dtc__max_depth=[None, 1, 2, 3, 4],
                  dtc__min_samples_split=[1, 2, 3, 4, 25],
                  dtc__class_weight=[None, 'balanced'],
                  dtc__random_state=[42]
                  )
# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split,StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Cross-validation for parameter tuning in grid search 
Stratified_Shuffle_Split = StratifiedShuffleSplit(
    labels_train,
    n_iter = 20,
    test_size = 0.5,
    random_state = 0
    )

# Create, fit, and make predictions with grid search
grid_search = GridSearchCV(pipeline,
                  param_grid=parameters,
                  scoring="f1",
                  cv=Stratified_Shuffle_Split,
                  error_score=0)
grid_search.fit(features_train, labels_train)
labels_predictions = grid_search.predict(features_test)

# Pick the classifier with the best tuned parameters
clf = grid_search.best_estimator_
print "\n", "Best parameters are: ", grid_search.best_params_, "\n"

# Print features selected and their importances
features_selected=[features_list[i+1] for i in clf.named_steps['feature_selection'].get_support(indices=True)]
scores = clf.named_steps['feature_selection'].scores_
importances = clf.named_steps['dtc'].feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
print 'The ', len(features_selected), " features selected and their importances:"
for i in range(len(features_selected)):
    print "feature no. {}: {} ({}) ({})".format(i+1,features_selected[indices[i]],importances[indices[i]], scores[indices[i]])

# Print classification report (focus on precision and recall)
report = classification_report( labels_test, labels_predictions )
print(report)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)