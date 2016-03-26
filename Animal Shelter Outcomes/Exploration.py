# -*- coding: utf-8 -*-

import pandas as pd
import sklearn as sk
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np


#Read Test Data
train_import = pd.read_csv("/Users/grazim/Documents/Kaggle/Train.csv", sep = ",")
train_import.shape

test_import = pd.read_csv("/Users/grazim/Documents/Kaggle/Test.csv", sep = ",")

#Check data types
train_import.dtypes

#Get summary statistics
train_import.describe()

#Latitude 0-(+/-90) Y
#Longitude 0-(+/-180) X

#Latitude of 90 would be far away from San Francisco.  Appears to be some messy data
test=train_import[train_import['Y'] == 90]

#Remove these extreme observations from the training data, set the values to the mean in the test set
extreme = train_import[train_import.Y == 90]
#67 observations at the north pole (hehe)
train_import = train_import[train_import.Y != 90]

#subset by index
#train_import = train_import.drop(extreme.index)

#Pull out time of day from the data as well as date
train= train_import
del train_import

train['event'] = 1

train['Date'] = pd.to_datetime(train.Dates)
train['Date'] = pd.DatetimeIndex(train.Dates)
#Pull hour out of datetime
train['Hour'] = pd.DatetimeIndex(train.Dates).hour
train['Year'] = pd.DatetimeIndex(train.Dates).year
train['Month'] = pd.DatetimeIndex(train.Dates).month
train['Week'] = pd.DatetimeIndex(train.Dates).week

#train['Month'] = train.Month.map("{:02}".format) #add leading 0
#train['YrMo'] = train.Year.map(str) + train.Month.map(str)
#train['YrMo'] = train.YrMo.astype(int)

train['Hour'] = train.Hour.astype(int)



#More crime appears to occur after, say around, 8 am
train.Hour.value_counts(sort = False, ascending = False).plot(kind='bar')

#Attempt to geocode each of the crimes.  Are there clusters of crimes in the data?

#Find map ranges
max_lat = train.Y.max()
max_long = train.X.max()
min_lat = train.Y.min()
min_long = train.X.min()




map = Basemap(llcrnrlon = min_long,
              llcrnrlat = min_lat,
              urcrnrlon = max_long,
              urcrnrlat = max_lat)

map.drawcoastlines()
map.drawcountries()
map.fillcontinents(color = 'coral')
map.drawmapboundary()
 
plt.show()




#Explore Day of week
train.DayOfWeek.value_counts(ascending = False).plot(kind = 'bar')
#Possibly more crimes on Friday

#Explore District
train.PdDistrict.value_counts().plot(kind = 'bar')
#The number of crimes in the southern part of town appears o be very high


YrWeek = train[['Week', 'Year', 'event']].groupby(['Year', 'Week']).count()
YrMo = train[['Month', 'Year', 'event']].groupby(['Year', 'Month']).count()
YrWeek.plot(kind='line')
YrMo.plot(kind='line')

hourly_district_events = train[['PdDistrict','Hour','event']].groupby(['PdDistrict','Hour']).count().reset_index()
hourly_district_events_pivot = hourly_district_events.pivot(index='Hour', columns='PdDistrict', values='event').fillna(method='ffill')
hourly_district_events_pivot.interpolate().plot(title='number of cases hourly by district', figsize=(10,6))

#
#hourly_Category_events = train[['Category','Hour','event']].groupby(['Category','Hour']).count().reset_index()
#hourly_Category_events_pivot = hourly_Category_events.pivot(index='Hour', columns='Category', values='event').fillna(method='ffill')
#hourly_Category_events_pivot.interpolate().plot(title='number of cases hourly by Category', figsize=(10,6))


#Are there commond intersections or streets occuring?
Unique_addresses = len(train.Address.unique())
#Over 28000 unique addresses

#Are certain crimes more or less likely to occur at a street corner? 
train['street_corner'] = train['Address'].apply(lambda x: 1 if '/' in x else 0)

#Plot street_corner by Category

train[['street_corner', 'Category']].groupby('Category').count().plot(kind = 'bar')
#A pure count is rather useuse because there is no comparative standard

#Look at the Percentatge of each crime in each category

crime_cat = sorted(train.Category.unique())


cat_counts = {}
cat_corner_counts = {}
cat_percent_corner = {}
#Hash each category name to its total count
for crime in crime_cat:
    crime_data = train[train.Category == crime]
    count = crime_data.shape[0]
    crime_data_street = crime_data[crime_data.street_corner ==1]
    count_street = crime_data_street.shape[0]
    cat_counts[crime] = count
    cat_corner_counts[crime] = count_street
    cat_percent_corner[crime] = round(100*(count_street/count),2)
    
    
corner_percent = pd.DataFrame.from_dict(cat_percent_corner, orient = "index")
corner_percent.columns = ["percent"]
corner_percent.plot(kind = 'bar', xlim = (0,100))




#Create dummy variables out of hour, day of week, month, and PdDistrict
train['Hour'] = train.Hour.astype(object)
train['Month'] = train.Month.astype(object)
dummies = pd.get_dummies(train[['Hour', 'Month', 'PdDistrict']])

#Normalize latitude (Y) and longitude (X)
train['x_norm'] = (train.X - train.X.mean())/train.X.std()
train['y_norm'] = (train.Y - train.Y.mean())/train.Y.std()




train_final = pd.concat([train[['Category','x_norm', 'y_norm']], dummies], axis = 1)

#Split data into train/test and start running models
from sklearn.cross_validation import train_test_split

training, validation = train_test_split(train_final, test_size = 0.2)


resp_training = training['Category']
pred_training = training.drop(['Category'], axis = 1)

resp_valid = validation['Category']
pred_valid = validation.drop(['Category'], axis = 1)

import pylab as pl
from sklearn import ensemble, metrics
RF = ensemble.RandomForestClassifier(n_estimators = 20)
RF_Fit = RF.fit(pred_training, resp_training)
RF_Pred = RF.predict(pred_valid)
RF_Pred_Prob = RF.predict_proba(pred_valid)

labels = crime_cat
RF_confuse = metrics.confusion_matrix(resp_valid, RF_Pred, labels)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(RF_confuse)
pl.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
pl.xlabel('Predicted')
pl.ylabel('True')
pl.show()

#Create Dummy variables from Hour
Hours = pd.get_dummies(train['Hour'])
Hours = pd.concat([train['Category'], Hours], axis = 1)

LogReg = sk.linear_model.LogisticRegression(solver = "lbfgs", multi_class = 'multinomial')
LogReg_Fit = LogReg.fit(pred_training,resp_training)
LogReg_Pred = LogReg.predict(pred_valid)
LogReg_Pred_Prob = LogReg.predict_proba(pred_valid)

RF_Report = metrics.classification_report(resp_valid, RF_Pred, labels = labels)


LogReg_Report = metrics.classification_report(resp_valid, LogReg_Pred, labels = labels)


#metrics.log_loss(resp_valid, pd.DataFrame(RF_Pred))


file = open('/Users/Grazim/Desktop/out.txt', 'w')
file.write(RF_Report)
file.write(LogReg_Report)

file.close()

def clean_data(data, test = False, output_resp = False):
    '''Will prepare data for input into models'''
    
    #Find extreme values in Lat and long and replace with the median value
    if test == True:    
        #Replace extreme values with median in test set
        data[data.X ==90] = data.X.median() 
    else:
        #Delete extreme values in validation set
        data = data[data.X != 90]        
    
    #Extrapolate date and time information
    data['Date'] = pd.DatetimeIndex(data.Dates)
    #Pull hour out of datetime
    data['Hour'] = pd.DatetimeIndex(data.Dates).hour
    #data['Year'] = pd.DatetimeIndex(data.Dates).year
    data['Month'] = pd.DatetimeIndex(data.Dates).month
    #data['Week'] = pd.DatetimeIndex(data.Dates).week
    
    data['Hour'] = data.Hour.astype(object)
    data['Month'] = data.Month.astype(object)

    #Street corner
    data['street_corner'] = data['Address'].apply(lambda x: 1 if '/' in x else 0)
    data['street_corner'] = data.street_corner.astype(object)    
    
    #create dummy variables
    dummies = pd.get_dummies(data[['Hour', 'Month', 'PdDistrict']])

    #Normalize latitude (Y) and longitude (X)
    data['x_norm'] = (data.X - data.X.mean())/data.X.std()
    data['y_norm'] = (data.Y - data.Y.mean())/data.Y.std()
    
    #Final Dataset
    if output_resp == False:
        data_out = pd.concat([data[['x_norm', 'y_norm']], dummies], axis = 1)
    else:
        data_out = pd.concat([data[['Category','x_norm', 'y_norm']], dummies], axis = 1)

    return(data_out)




train = clean_data(train_import, output_resp = True)
del train_import
del test_import

import pylab as pl
from sklearn import ensemble, metrics
RF = ensemble.RandomForestClassifier(n_estimators = 100)
RF_Fit = RF.fit(train.drop(['Category'], axis = 1), train['Category'])

var_import = RF.feature_importances_
pred = RF.estimators_
var_import = pd.DataFrame(var_import, index = classes)

features = train.drop(['Category'], axis = 1).columns
importances = RF.feature_importances_
indices = np.argsort(importances)

plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')



test = clean_data(test_import, test = True)
test_pred_prob = pd.DataFrame(RF.predict_proba(test))


#Prepare predictions file for export
test_pred_prob.to_csv(path_or_buf = "/Users/Grazim/documents/Kaggle/test_pred.csv",header = RF.classes_ )
	








