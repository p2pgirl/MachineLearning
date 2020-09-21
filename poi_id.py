#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from time import time
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn import metrics
  
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


#global variables
employee_list = list()
finance_features = []
poi = 'poi'

####GO TO MAIN BELOW FOR REST OF PROCESSING...



### FUNCTIONS!!!! **********************************************************
#Use KBestFeatures to find the 10 best features to use  
def best_features_suggestion(data_dict, most_features_list):
    data = featureFormat(data_dict, most_features_list)
    labels, features = targetFeatureSplit(data)
    k_best = SelectKBest(k=10)
    k_best.fit(features, labels)
    scores = k_best.scores_
    scores_tuples = zip(most_features_list[1:], scores)
    scores_tuples_sorted = list(reversed(sorted(scores_tuples, key=lambda x: x[1])))
    print "K Best Features: ", scores_tuples_sorted
    

def employee_listing():
    temp_emp_list = list()
    for employee, category in data_dict.items():
        temp_emp_list.append(employee)
    return temp_emp_list

###PREPROCESSING/DATA EXPLORE task - explore the data
def dataexplore(employee_list):
    
    set_of_categories = set()
    
    no_of_employees = 0
    poi_count_emp = 0
    email_count_null = 0
    salary_count_null = 0
    bonus_count_null = 0
    exer_stock_null = 0
    total_payments_null = 0
    
    for employee, category in data_dict.items():
        #print 'Employee Name:', employee
        no_of_employees += 1
        
        #How many POIs are there in the dataset?
        if data_dict[employee]["poi"]==True:
            poi_count_emp += 1
            
        #How many people have a salary?
        #What about an email address?         
        if data_dict[employee]["email_address"] == "NaN":
            email_count_null += 1
        if data_dict[employee]["salary"] == "NaN":
            salary_count_null += 1
        if data_dict[employee]["bonus"] == "NaN":
            bonus_count_null += 1
        if data_dict[employee]["exercised_stock_options"] == "NaN":
            exer_stock_null += 1
        if data_dict[employee]["total_payments"] == "NaN":
            total_payments_null += 1
    
    
            
        ##For each person, how many features are available? 
        for individual_category in category:
            no_of_categories = len(category)
            set_of_categories.add(individual_category)
    employee_list.sort()
    
    print
    print "INFORMATION ABOUT DATA"           
    print 'Total Employee Number: ', no_of_employees
    print 'Total Number of categories (features):  ', no_of_categories
    print 'Number of poi (people of interest: ', poi_count_emp
    print 'Number of Employees with an Email ', no_of_employees - email_count_null
    print 'Number of Employees with an Email of NaN: ', email_count_null
    print 'Number of Employees with a Salary ', no_of_employees - salary_count_null
    print 'Number of Employees with a Salary of NaN: ', salary_count_null
    print 'Number of Employees with a Bonus: ', no_of_employees - bonus_count_null
    print 'Number of Employees with a Bonus of NaN: ', bonus_count_null
    print 'Number of Employees with Exercised Stock Options: ', \
        no_of_employees - exer_stock_null
    print 'Number of Employees with Stock Options of NaN: ', exer_stock_null
    print 'Number of Employees with Payments: ', no_of_employees - total_payments_null
    print 'Number of Employees NaN Payments', total_payments_null
    print
    
    
    category_names = ['Missing Data', 'Data Supplied']    
    
    results = {
    'Employee Emails': [email_count_null, no_of_employees - email_count_null],
    'Employee Salary': [salary_count_null, no_of_employees  - salary_count_null],
    'Employee Bonus': [bonus_count_null, no_of_employees - bonus_count_null],
    'Employee Exercised Stock Opt': [exer_stock_null, no_of_employees - exer_stock_null],
    'Employee with Payments': [total_payments_null, no_of_employees - total_payments_null]
    }
    
    survey(results, category_names)
    plt.show()
    
    #What percentage of people in the dataset as a whole is this? 
    percentage_with_payments = (float(total_payments_null)/no_of_employees) * 100
    print "Percentage of Employees with Payments: %.2f" % percentage_with_payments + "%"
    print
    
    #This section is if information is needed about a particular Employee.
    ## Select "n" if you do not wish to do this
    print
    print 'EMPLOYEE INQUIRY SECTION'    
    print
    question = "n"
    
    '''
    NOTE:  UNCOMMENT THE BELOW QUESTION TO ENQUIRE ABOUT AN EMPLOYEE SPECIFICALLY
    '''
    ###question = raw_input("Do you want to inquire about an employee? y/n:  " ).lower()
    if question == "n":
        print "Exiting inquiry..."
    elif question == "y":
        print "EMPLOYEE NAME LIST: "
        print employee_list
        employeeFName = str(raw_input("Enter employee first name: ")).upper()
        employeeLName = str(raw_input("Enter emplyoyee last name:  ")).upper()
        employeeMInitial = str(raw_input\
                               ("Enter employee middle initial, if none type \"NONE\":  ")).upper()
        if employeeMInitial != "NONE":
            employeeFullName = employeeLName + " " + employeeFName + " " + \
            employeeMInitial
        else:
            employeeFullName = employeeLName + " " + employeeFName
        if data_dict[employeeFullName]:
            print "What features would you like to see? (select from list above)"
            print set_of_categories
            feature_to_see = str(raw_input("Input wanted feature:  "))
            answer_employee_question = data_dict[employeeFullName][feature_to_see]
            print 
            print "The answer is:  ", answer_employee_question
            print
        else:
            print "No such employee: %s" % employeeFullName
    else: 
        print "Incorrect response type"

###END: dataexplore()    

def survey(results, category_names):
    """
    displays plot for employee data
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c)), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax

###
### TASK 2: Remove outliers
def outlier_clean(features, employee_list):

    finance_features = features
    
    #Clean the data: remove NaNs and less than zeros for financial data being used
    print "Cleaning financial data..."
    for employee, category in data_dict.items():
        for feature in finance_features:
            if data_dict[employee][feature] == "NaN" or \
                data_dict[employee][feature] < 0:
                    data_dict[employee][feature] = 0
    print
    print "DONE - Cleaning Data."
    print
    
    #Create plot to find any finance outliers
    features_outliers = ["bonus", "salary"]
    data = featureFormat(data_dict, features_outliers, remove_any_zeroes=True)
    target, features = targetFeatureSplit( data )
    
    #Finding outlier to remove
    salary_max = 0
    ##list variable to find biggest 3 salaries
    salary_list = []
    
    #data is bonus - 0 and salary - 1
    for point in data:
        salary = point[1]
    
        #determine greatest salary
        if salary >= salary_max:
            salary_max = salary
            salary_list.append(salary)
        
        
        bonus = point[0]
        plt.scatter( bonus, salary )
    
    plt.xlabel("bonus")
    plt.ylabel("salary")
    plt.show()
    
        ###Look at top 3 salaries
    salary_list.sort()
    salary_list.reverse()
    max_salary1 = salary_list[0]
    max_salary2 = salary_list[1]
    max_salary3 = salary_list[2]
    print "Top 3 salaries: ", max_salary1, max_salary2, max_salary3
    print
    #To find key associated with salary
    for employee, feature in data_dict.items():
        for feature, value in feature.items():
            if feature == "salary" and value == salary_max:
                print
                print "Biggest Salary! Name of employee: ", employee
                print
            if feature == "salary" and value == max_salary2:
                print
                print "2nd Biggest Salary: ", max_salary2
                print "2nd Biggest Belongs to:  ", employee
                print
            if feature == "salary" and value == max_salary3:
                print "3rd Biggest Salary: ", max_salary3
                print "3rd Biggest Belongs to:  ", employee
                print
        #######DONE finding key associated with salary
    
    print
    print "Now remove outlier named:  TOTAL..." 
    ### there's an outlier--remove it! 
    data_dict.pop("TOTAL", 0)
    print
    
    outlier_check = 0
    for employee, feature in data_dict.items():
        if employee == "TOTAL":
            outlier_check += 1
    if outlier_check == 0:
        print "Outlier verified removed."
        
        
    #"TOTAL" was an odd name.  I will check the list of names for any other 
    ##remainging outliers
    print "Verifying any other name oddities:  ", employee_list
    print
    #Now I will remove the other oddity I found
    print "Removing name \"The Travel Agency in the Park\..."
    data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
    print
    
###END: outlier_clean(finance_features)   


###
### TASK 3: Create new features

##TOOL used in computeFraction function
#  NaN will never equal itself, so if it's NaN the return statement will return 
#    False.  
def isNotNaN(testnum):
    return testnum == testnum

###END: isNotNaN()

##TOOL used in new_features function  
def computeFraction(poi_messages, all_messages):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """

    
    fraction = 0.
   
    if isNotNaN(all_messages) and isNotNaN(poi_messages):
        fraction = float(poi_messages) / float(all_messages)
    
    if isNotNaN(fraction) == False:
        fraction = 0.

    return fraction

###END: computeFraction()

def new_features(data_dict):
    for employee in data_dict:
    
        data_point = data_dict[employee]
    
        from_poi_to_this_person = data_point["from_poi_to_this_person"]
        to_messages = data_point["to_messages"]
        fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
        data_point["fraction_from_poi"] = fraction_from_poi
    
    
        from_this_person_to_poi = data_point["from_this_person_to_poi"]
        from_messages = data_point["from_messages"]
        fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
        data_point["fraction_to_poi"] = fraction_to_poi  

###END: new_features()
        
def plot_features_list_test():
    d = {'No. of Features': [2, 4, 7, 8, 9, 11], 
     'F1 Accuracy': [0.44, 0.18, 0.44, 0.5, 0.8, 0.67]}

    df = pd.DataFrame(data=d)
    
    plt.plot( 'No. of Features', 'F1 Accuracy', data=df, color='orange')
      
    # Add titles
    plt.title("Adaboost Classifier with Decision Tree", 
              loc='left', fontsize=12, fontweight=0, color='red')
    plt.xlabel("No. of Features")
    plt.ylabel("F1 Accuracy")
    
    plt.show()
    

### TASK 4: Try a varity of classifiers
def classifiers_try(features_train, features_test, labels_train, labels_test):
    #### import the sklearn modules
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier

    
    ### create classifier
    classifiers = [GaussianNB(), 
                   SVC(kernel="rbf", C=10000.0),
                   DecisionTreeClassifier(),   
                   DecisionTreeClassifier(min_samples_split = 40),
                   DecisionTreeClassifier(min_samples_split=40, max_depth=4),
                   AdaBoostClassifier(), 
                   AdaBoostClassifier(
                        DecisionTreeClassifier(min_samples_split=20, max_depth=4),
                        algorithm="SAMME.R",
                        n_estimators=60,
                        random_state=0)
                   ]
    
    for clf in classifiers:
        t0 = time()
        clf.fit(features_train, labels_train)
        time_info = round(time()-t0, 3)
        print
        print "***************************************"
        print "Classifier: %s --> " % clf
        print
        print "training time:", time_info, "s"
        print
        t0 = time()
        pred = clf.predict(features_test)
        time_info = round(time()-t0, 3)
        print "prediction time:", time_info, "s"
        print
        print "accuracy %.3f: " % accuracy_score(labels_test, pred)
        print
        print "f1 score:", metrics.f1_score(labels_test, pred)
        print 
        print "***************************************"
        print

###END: classifiers_try()
        
### TASK 5: Tune your classifier to achieve better than .3 precision and recall
def evaluate(labels_test, pred):

    
    print "accuracy %.3f: " % accuracy_score(labels_test, pred)
    print "f1 score:", metrics.f1_score(labels_test, pred)
    print
    poi_count = 0
    for poi_decider in labels_test:
        if poi_decider == 1:
            poi_count += 1
    print "How many POI's are in the test set:  ", poi_count
    no_in_testSet = len(features_test)
    print "How many people in the test set total? ", no_in_testSet
    
    matched = 0
    for i in range(no_in_testSet):
        predicted_data_point = pred[i]
        actual_data_point = labels_test[i]
        if (predicted_data_point == actual_data_point) and predicted_data_point == 1:
            matched += 1
    print "Number of true positives: ", matched
    print "Recall: %.3f " % metrics.recall_score(labels_test, pred)
    print "Precision: %.3f " % metrics.precision_score(labels_test, pred)
    print "f1 score: %.3f " % metrics.f1_score(labels_test, pred)
    print 
    report = classification_report(labels_test, pred, target_names=['NON_POI', 'POI'])
    print "Classification Report:  ", report
    print "***************************************"
    print

###END: evaluate(features_train, features_test, labels_train, labels_test)
        
### END FUNCTIONS!!!! ******************************************************

###  MAIN CODE *************************************************************

###TASK 1: Select the features being used.
most_features_list = ['poi','salary', 'bonus', \
                 'exercised_stock_options', 'from_messages', \
                     'from_poi_to_this_person', \
                         'from_this_person_to_poi', \
                             'from_messages', 'to_messages', \
                                 'restricted_stock','shared_receipt_with_poi', \
                                     'total_payments', 'expenses', \
                                        'loan_advances', 'total_stock_value'] 

best_features_suggestion(data_dict, most_features_list)

#this list is based on the function-> best_features_suggestion   
features_list = ['poi','salary', 'bonus', \
                 'exercised_stock_options', \
                     'total_payments', 'loan_advances', \
                         'shared_receipt_with_poi', \
                             'from_poi_to_this_person', \
                                 'from_this_person_to_poi'] 

#TASK 2 - REMOVE OUTLIERS
employee_list = employee_listing()
dataexplore(employee_list)
finance_features = ['salary', 'bonus', \
                    'exercised_stock_options', 'total_payments', \
                        'loan_advances']
outlier_clean(finance_features, employee_list)


#TASK 3 - CREATE NEW FEATURES
new_features(data_dict)
features_list.append('fraction_from_poi')
features_list.append('fraction_to_poi')
print "REVIEW features_list: ", features_list
print

#Re-Review 10 best features to remove.  I'm wondering if the from and to 
#emails are not needed
best_features_suggestion(data_dict, features_list)

#decided to remove two fields:  ('from_poi_to_this_person', 5.041257378669385)
##  and ('from_this_person_to_poi', 2.295183195738003).  Fractions above will
##  suffice and "best_features" rates them low.
features_list.remove('from_poi_to_this_person')
features_list.remove('from_this_person_to_poi')


#Used to test overall F1 score with classifier depending on feature number

## 7 features left
#features_list.remove('fraction_from_poi')
##6 features left
#features_list.remove('loan_advances')
#features_list.remove('shared_receipt_with_poi')
##4 features left
#features_list.remove('total_payments')
#features_list.remove('fraction_to_poi')
##2 features left
#features_list.remove('salary')
#features_list.remove('bonus')


#plot the features test result
plot_features_list_test()

my_dataset = data_dict
my_feature_list = features_list


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_feature_list, sort_keys = True)

labels, features = targetFeatureSplit(data)


features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

classifiers_try(features_train, features_test, labels_train, labels_test)

print
print "***********************************"
print "Best classifier is Adaboost and Decision Tree added"
print "The reason to add DT manually was to narrow down Adaboost's" 
print "default DT parameters"
print "***********************************"
print



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#### import the sklearn modules
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA


clf = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=20, max_depth=4),
                        algorithm="SAMME.R",
                        n_estimators=60,
                        random_state=0)
               


###Using PCA with AdaBoost

#my_steps = [('pca', PCA(n_components = 3) ),
#            ('clf', clf)]
#clf = Pipeline(my_steps)


###Using MinMaxScaler with AdaBoost
#from sklearn.preprocessing import MinMaxScaler
#my_steps = [('scaler', MinMaxScaler() ),
#            ('clf', clf)]
#clf = Pipeline(my_steps)



print "************************************"
print "MinMaxScaler unneeded for AdaBoost and dataset"
print "   (see report for more info)       "
print "************************************"
print

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

evaluate(labels_test, pred)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, my_feature_list)
