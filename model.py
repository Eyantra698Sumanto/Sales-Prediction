import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import labels as lb

# Reading the training data
train = pd.read_csv('dataset/train.csv')

# Data Cleaning in Training DataFrame
train['city'].replace('ASHEVILLE', 'ASHVILLE', inplace=True)

# List of columns with their unique value in the Training DataFrame
city_list = sorted(train.city.unique())
store_location_list = sorted(train.store_location.unique())
location_employee_code_list = sorted(train.location_employee_code.unique())
credit_score_list = sorted(train.credit_score.unique())
credit_score_range_list = sorted(train.credit_score_range.unique())

# Labelled values of lists
city_labels = lb.assign_labels(city_list)
store_location_labels = lb.assign_labels(store_location_list)
location_employee_code_labels = lb.assign_labels(location_employee_code_list)
credit_score_labels = lb.assign_labels(credit_score_list)
credit_score_range_labels = lb.assign_labels(credit_score_range_list)

# Updating DataFrame
lb.update_dataframe(train, 'city', city_labels)
lb.update_dataframe(train, 'store_location', store_location_labels)
lb.update_dataframe(train, 'location_employee_code', location_employee_code_labels)
lb.update_dataframe(train, 'credit_score', credit_score_labels)
lb.update_dataframe(train, 'credit_score_range', credit_score_range_labels)

# Deleting unused columns
df_train = train.drop('state', axis=1)
df_train = df_train.drop('zip', axis=1)
df_train = df_train.drop('time_zone', axis=1)
df_train = df_train.drop('latitude', axis=1)
df_train = df_train.drop('longitude', axis=1)
df_train = df_train.drop('total_sales', axis=1)

# Extracting the features
X = df_train.loc[:, 'business_type':'actual_credit_score']
Y = train.loc[:, 'total_sales']

# Training the model

clf_tree = DecisionTreeClassifier()
clf_dec = clf_tree.fit(X, Y)
clf_knn = KNeighborsClassifier(n_neighbors=5)
clfKNN = clf_knn.fit(X, Y)


'''
    Here comes  the testing dataset
'''


# Testing DataFrame
test = pd.read_csv('dataset/test.csv')

# Data Cleaning in Testing DataFrame
test['city'].replace('ASHEVILLE', 'ASHVILLE', inplace=True)

# List of columns with their unique value in the Training DataFrame
city_list_test = sorted(test.city.unique())
store_location_list_test = sorted(test.store_location.unique())
location_employee_code_list_test = sorted(test.location_employee_code.unique())
credit_score_list_test = sorted(test.credit_score.unique())
credit_score_range_list_test = sorted(test.credit_score_range.unique())

# Labelling the test dataset
city_labels = lb.assign_labels_test(city_list, city_list_test, city_labels)
store_location_labels = lb.assign_labels_test(store_location_list, store_location_list_test, store_location_labels)
location_employee_code_labels = lb.assign_labels_test(location_employee_code_list, location_employee_code_list_test, location_employee_code_labels)
credit_score_labels = lb.assign_labels_test(credit_score_list, credit_score_list_test, credit_score_labels)
credit_score_range_labels = lb.assign_labels_test(credit_score_range_list, credit_score_range_list_test, credit_score_range_labels)

# Updating DataFrame
lb.update_dataframe(test, 'city', city_labels)
lb.update_dataframe(test, 'store_location', store_location_labels)
lb.update_dataframe(test, 'location_employee_code', location_employee_code_labels)
lb.update_dataframe(test, 'credit_score', credit_score_labels)
lb.update_dataframe(test, 'credit_score_range', credit_score_range_labels)

# Deleting unused columns
df_test = test.drop('state', axis=1)
df_test = df_test.drop('zip', axis=1)
df_test = df_test.drop('time_zone', axis=1)
df_test = df_test.drop('latitude', axis=1)
df_test = df_test.drop('longitude', axis=1)
df_test = df_test.drop('outlet_no', axis=1)


prediction = clf_tree.predict(df_test)
prediction1 = clf_knn.predict(df_test)

# Results
print('Decision Tree')
print(prediction)

print('kNN')
print(prediction1)

# Creating the csv file for submitting the solution
submission = pd.DataFrame({
        "outlet_no": test['outlet_no'],
        "total_sales_actual": prediction
    })

submission.to_csv("solution.csv", index=False)

