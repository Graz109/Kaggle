import sklearn as sk
from sklearn import ensemble
import pandas as pd
import operator

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

def data_import(filepath):
    
    data = pd.read_csv(filepath)    
    
    #fixed or not
    fixed_sex = data['SexuponOutcome'].str.split(' ')
    #If spayed or neutered, make fixed 1, else false
    data['fixed'] = [1 if x in ['Neutered', 'Spayed'] else 0 for x in fixed_sex.str[0]]
    
    #Sex
    data['sex'] = fixed_sex.str[1]
    
    #Age
    def calc_age_in_years(x):
        x = str(x)
        if x == 'nan': return 0
        age = int(x.split()[0])
        if x.find('year'): 
            return age 
        elif x.find('month'): 
            return age / 12
        elif x.find('week'): 
            return age / 52
        elif x.find('days'): 
            return age / 365
        else: 
            return 0
        
    #NOTE: All credit for this function goes to Kaggle user Eugenia Uchaeva!
    data['age_years'] = data.AgeuponOutcome.apply(calc_age_in_years)
    #data['age_years'] = data['age_years'].astype(str)
    
    #Time of Outcome
    #Convert DateTime to day of week, time of day
    data['DateTime'] = pd.to_datetime(data.DateTime)
    #data['year'] = data['DateTime'].dt.year
    data['month'] = data['DateTime'].dt.month
    data['dayofweek'] = data['DateTime'].dt.dayofweek # Monday = 0, Sunday = 6
    data['hour'] = data['DateTime'].dt.hour
    #data['weekofyear'] = data['DateTime'].dt.weekofyear
    
    #Convert to catagorical variables
    #data['year'] = data.year.astype(str)
    data['month'] = data.month.astype(str)
    data['dayofweek'] = data.dayofweek.astype(str)
    data['hour'] = data.hour.astype(str)
    #data['weekofyear'] = data.weekofyear.astype(str)  
        
    #Pull out mixed breed or mut
    def Breed_Type(x):
        x = str(x)
        if x.find('Mix') >= 0:
            return 'Unknown Mix'
        elif x.find('/') >= 0:
            return 'Known Mix'
        else:
            return 'Pure'

    data['breed_category'] = data.Breed.apply(Breed_Type)
    
    
    #Create dummy variables out of categorical variables
    
    data = pd.concat([data, pd.get_dummies(data[['sex','month','dayofweek','hour','breed_category']])], axis = 1)
    
    
    #Drop unneeded columns
    data = data.drop(['DateTime','SexuponOutcome','AgeuponOutcome','breed_category','Name','hour', 'month', 'dayofweek','sex'], axis = 1)
    
    
    return(data)
    
train = data_import("/Users/grazim/Documents/Kaggle_Local/Shelter Animal Outcomes/train.csv")
test = data_import("/Users/grazim/Documents/Kaggle_Local/Shelter Animal Outcomes/test.csv")


age = train['age_years']
age_test = test['age_years']

pred = train.drop(['AnimalID', 'OutcomeType', 'OutcomeSubtype', 'AnimalType', 'Breed', 'Color','age_years'], axis = 1)
pred = pd.concat([age, pred], axis=1)

test_pred = test.drop(['ID', 'AnimalType', 'Breed', 'Color', 'age_years'], axis = 1)
test_pred = pd.concat([age_test, test_pred], axis = 1)

resp = train['OutcomeType']

#Random Forest
RF = ensemble.RandomForestClassifier(n_estimators = 10000, n_jobs = -1, verbose = 10)
RF.fit(pred, resp)


def Feature_Importance_Plot(pred, fit_object):
    '''   #Plot feature importances
        pred: list of predictors
        fit_object: the object that contains model information
    '''
    FI = fit_object.feature_importances_

    feat_imp = pd.Series(FI, pred).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances', figsize = (10,10))
    plt.ylabel('Feature Importance Score')



test_predictions = pd.DataFrame(RF.predict_proba(test_pred))

test_predictions.index += 1
#Output predictions
test_predictions.to_csv("/Users/grazim/Documents/Kaggle_Local/Shelter Animal Outcomes/RF_pred.csv",header = RF.classes_ )


pred = pred.drop(['hour_5'], axis = 1)
test_pred = test_pred.drop(['hour_3'], axis = 1) 

#XGBoost
XGB = XGBClassifier(n_estimators=15000)
XGB.fit(pred, resp)

test_predictions = pd.DataFrame(XGB.predict_proba(test_pred))
test_predictions.index += 1


test_predictions.to_csv("/Users/grazim/Documents/Kaggle_Local/Shelter Animal Outcomes/XGB_pred.csv",header = XGB.classes_ )


output = pd.DataFrame(pred.columns)
output.to_csv("/Users/grazim/Documents/Kaggle_Local/Shelter Animal Outcomes/pred.csv")

test_output = pd.DataFrame(test_pred.columns)
test_output.to_csv("/Users/grazim/Documents/Kaggle_Local/Shelter Animal Outcomes/test_pred.csv")




