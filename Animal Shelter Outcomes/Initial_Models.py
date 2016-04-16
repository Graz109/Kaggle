import sklearn as sk
from sklearn import ensemble
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from xgboost.sklearn import XGBClassifier

def data_import(filepath):
    
    data = pd.read_csv(filepath)    
    
    #fixed or not
    fixed_sex = data['SexuponOutcome'].str.split(' ')
    #If spayed or neutered, make fixed 1, else false
        
    #data['fixed'] = [1 if x in ['Neutered', 'Spayed'] else 0 for x in fixed_sex.str[0]]
    #data['unknown_fixed'] = [1 if x in ['Unknown'] else 0 for x in fixed_sex.str[0]]
    #data['active_sex'] = [1 if x in ['Intact'] else 0 for x in fixed_sex.str[0]]

    data['fixed'] = [1 if x in ['Neutered', 'Spayed'] else 0 for x in fixed_sex.str[0]]
    
    #Sex
    #data['male'] = [1 if x in ['Male'] else 0 for x in fixed_sex.str[1]]
    #data['female'] = [1 if x in ['Female'] else 0 for x in fixed_sex.str[1]]
    #data['unknown_sex'] = [1 if x in ['Unknown'] else 0 for x in fixed_sex.str[0]]
    
    data['sex'] = fixed_sex.str[1]
    data['sex'] = [1 if x in['Male'] else 0 for x in data.sex]
    
    data['unknown_sex_fixed'] = [1 if x in ['Unknown'] else 0 for x in fixed_sex.str[0]]    
    
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
        
    def calc_age_in_days(x):
        x = str(x)
        if x == 'nan': return 0
        age = int(x.split()[0])
        if x.find('year'): 
            return age*364 
        elif x.find('month'): 
            return age*30
        elif x.find('week'): 
            return age*7
        elif x.find('days'): 
            return age
        else: 
            return 0
        
    #NOTE: All credit for this function goes to Kaggle user Eugenia Uchaeva!
    data['age_years'] = data.AgeuponOutcome.apply(calc_age_in_years)
    #data['age_days'] = data.AgeuponOutcome.apply(calc_age_in_days)        
    
    #Time of Outcome
    #Convert DateTime to day of week, time of day
    data['DateTime'] = pd.to_datetime(data.DateTime)
    data['year'] = data['DateTime'].dt.year
    data['month'] = data['DateTime'].dt.month
    data['dayofweek'] = data['DateTime'].dt.dayofweek # Monday = 0, Sunday = 6
    data['hour'] = data['DateTime'].dt.hour
    #data['weekofyear'] = data['DateTime'].dt.weekofyear
    
    data['minutes'] = data['hour']*60 + data['DateTime'].dt.minute    
    
#    days_month = []
#    for i in range(0,len(data)):
#        days_month.append(monthrange(data['year'][i], data['month'][i])[1])
#    data['days_month'] = days_month
    
    #data['days_in_month'] = np.asarray(map(lambda x,y: monthrange(x,y)[1], data['year'].values, data['month'].values))    
    
    #Convert to catagorical variables
    data['year'] = data.year.astype(str)
    data['month'] = data.month.astype(str)
    data['dayofweek_num'] = data.dayofweek
    data['dayofweek'] = data.dayofweek.astype(str)
    data['hour'] = data.hour.astype(str)
    #data['weekofyear'] = data.weekofyear.astype(str)  
        
        
    #AM/PM: AM-[0-->12), PM-[12,24]
    #data['AM'] = [1 if x in ['0','1','2','3','4','5','6','7','8','9','10','11'] else 0 for x in data.hour]
        
    #Seasons
#    def seasons(x):
#        if x in ['3','4','5']:
#            return('Spring')
#        elif x in ['6','7','8']:
#            return('Summer')
#        elif x in ['9','10','11']:
#            return('Autumn')
#        elif x in ['12','1','2']:
#            return('Winter')
#        
#    data['Season'] = data.month.apply(seasons)      
        
    #data['weekend'] = [1 if x in ['5','6'] else 0 for x in data.dayofweek]
    #Color
    data['color_length'] = [len(x) for x in data.Color]        
    data['single_color']= [0 if x.find('/') >= 0 else 1 for x in data.Color] #Terrible feature
    
    #Named or not
    data['named'] = [1 if x == False else 0 for x in pd.isnull(data.Name)]        
    
    #Pull out mixed breed or mut
    def Breed_Type(x):
        x = str(x)
        if x.find('Mix') >= 0:
            return 'Unknown Mix'
        elif x.find('/') >= 0:
            #return 'Known Mix'
            return 'Unknown Mix'
        elif x.find('Domestic') >=0:
            return 'Unknown Mix'
        else:
            return 'Pure'

    data['breed_category'] = data.Breed.apply(Breed_Type)
    
    
    #Create dummy variables out of categorical variables
    
    data = pd.concat([data, pd.get_dummies(data[['month','dayofweek', 'year','hour']])], axis = 1)
    #'dayofweek','hour', 'month', 'year', 'breed_category'
    
    #data['Breed_len'] = [len(x) for x in data.Breed]    
    data['Breed_words'] = [len(x.replace('/', ' ').replace('_', ' ').split(' ')) for x in data.Breed]        
    
    #Drop unneeded columns
    data = data.drop(['dayofweek', 'month', 'year','hour','DateTime','SexuponOutcome','AgeuponOutcome','Breed', 'breed_category', 'Name'], axis = 1)
    #'hour'
    
    return(data)
    
train = data_import("/Users/grazim/Documents/Kaggle_Local/Shelter Animal Outcomes/train.csv")
test = data_import("/Users/grazim/Documents/Kaggle_Local/Shelter Animal Outcomes/test.csv")


age = train['age_years']
age_test = test['age_years']

#age = train['age_days']
#age_test = test['age_days']

pred = train.drop(['AnimalID', 'OutcomeType', 'OutcomeSubtype', 'AnimalType','age_years'], axis = 1)
pred = pd.concat([age, pred], axis=1)

test_pred = test.drop(['ID','AnimalType','age_years'], axis = 1)
test_pred = pd.concat([age_test, test_pred], axis=1)

#Check for multicollinearity
#pred_mat = np.matrix(pred)
#EV = np.linalg.eig(pred.T * pred)
#Condition Number
#CN = np.sqrt( EV.max() / EV.min() )
#print('Condition No.: {:.5f}').format( CN )



#data = pd.read_csv("/Users/grazim/Documents/Kaggle_Local/Shelter Animal Outcomes/train.csv")    
#data['DateTime'] = pd.to_datetime(data.DateTime)
#data['year'] = data['DateTime'].dt.year
#data['month'] = data['DateTime'].dt.month
#
#lam = lambda year, month: monthrange(year, month)[1]
#test= data['days_in_month'] = map(lam, data['year'].values, data['month'].values)
#
#test1 = data.apply(monthrange, args = (data.year, data.month), axis = 0)
#
#
#a = data.apply(lam, (2013, 4))
#
#a = data[['year','month']].apply(monthrange)



#Function to break color combinations 
def color_split(x):
    #x = str(x)
    x = [item.replace(" ", "_") for item in x]
    x = [item.split('/') for item in x]
    #x = [item.split(' ') for item in x]
    return(x)

colors = train.Color.unique()        
colors = color_split(colors)
colors = [item for sublist in colors for item in sublist]

unique_colors = Counter(colors)
unique_colors = pd.DataFrame.from_dict(unique_colors, orient = 'index')
unique_colors = unique_colors.sort_index()
#[item for sublist in l for item in sublist]

#It appears I can focus on major color groups such as  Black, Blue, Brown, Calico, Chocolate, Cream, Red, orange, Tan, White, Yellow

#Try to add just some basic colors first (Black, Blue, Brown, Red, White)

pred['black'] = [1 if x.find('Black') >=0 else 0 for x in train.Color]
pred['blue'] = [1 if x.find('Blue') >=0 else 0 for x in train.Color]
pred['brown'] = [1 if x.find('Brown') >=0 else 0 for x in train.Color]
pred['tan'] = [1 if x.find('Tan') >=0 else 0 for x in train.Color]
pred['white'] = [1 if x.find('White') >=0 else 0 for x in train.Color]
pred['tabby'] = [1 if x.find('Tabby') >=0 else 0 for x in train.Color]


test_pred['black'] = [1 if x.find('Black') >=0 else 0 for x in test.Color]
test_pred['blue'] = [1 if x.find('Blue') >=0 else 0 for x in test.Color]
test_pred['brown'] = [1 if x.find('Brown') >=0 else 0 for x in test.Color]
test_pred['tan'] = [1 if x.find('Tan') >=0 else 0 for x in test.Color]
test_pred['white'] = [1 if x.find('White') >=0 else 0 for x in test.Color]
test_pred['tabby'] = [1 if x.find('Tabby') >=0 else 0 for x in test.Color]

pred = pred.drop(['Color'], axis=1)
test_pred = test_pred.drop(['Color'], axis = 1)










resp = train['OutcomeType']



##Tune RF
#from sklearn import ensemble, metrics, decomposition, grid_search, preprocessing
#
#param_grid1 = {'max_depth':list(range(50,100,5))
#              #,'min_samples_split': list(range(500,1000,100))
#              }
#gsearch = grid_search.GridSearchCV(estimator = ensemble.RandomForestClassifier(n_estimators = 1000, n_jobs = -1, verbose = 1),
#                                   param_grid = param_grid1, n_jobs = -1)
#                                   
#gsearch.fit(pred, resp)
#gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_

#Remove low importance features
pred = pred.drop(['hour_5','hour_22','hour_21','hour_6','hour_23','hour_20','hour_7','hour_8','hour_10','hour_19','hour_0'], axis = 1)    
test_pred = test_pred.drop(['hour_22','hour_21','hour_6','hour_23','hour_20','hour_7','hour_8','hour_10','hour_19','hour_0'], axis = 1)    
#0.65995734969508768

#Random Forest
RF = ensemble.RandomForestClassifier(n_estimators = 10000, n_jobs = -1, verbose = 1, oob_score = True)
RF.fit(pred, resp)


RF.oob_score_

#oob_list = []
#min = int(np.sqrt(pred.shape[1]).round(1))
#for i in range(min, min + 15):
#    RF = ensemble.RandomForestClassifier(n_estimators = 5000, n_jobs = -1, verbose = 10, oob_score = True, max_features = i)
#    RF.fit(pred, resp)
#    oob = RF.oob_score_
#    oob_list.append([i, oob])
#    print("Complete with ", i)
#    

## words in breed: 0.65464476785513859
def Feature_Importance_Plot(pred, fit_object):
    '''   #Plot feature importances
        pred: list of predictors
        fit_object: the object that contains model information
    '''
    FI = fit_object.feature_importances_

    feat_imp = pd.Series(FI, pred).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances', figsize = (10,10))
    plt.ylabel('Feature Importance Score')

Feature_Importance_Plot(pred.columns, RF)



#Explore color
data = pd.read_csv("/Users/grazim/Documents/Kaggle_Local/Shelter Animal Outcomes/train.csv")    
unique_colors = data.Color.unique()
len(unique_colors)





test_predictions = pd.DataFrame(RF.predict_proba(test_pred))

test_predictions.index += 1
#Output predictions
test_predictions.to_csv("/Users/grazim/Documents/Kaggle_Local/Shelter Animal Outcomes/RF_pred.csv",header = RF.classes_ )





pred = pred.drop(['hour_5'], axis = 1)
test_pred = test_pred.drop(['hour_3'], axis = 1) 

#XGBoost
XGB = XGBClassifier(n_estimators=15000)
XGB.fit(pred, resp)



test_predictions_XGB = pd.DataFrame(XGB.predict_proba(test_pred))
test_predictions_XGB.index += 1



test_predictions.to_csv("/Users/grazim/Documents/Kaggle_Local/Shelter Animal Outcomes/XGB_pred.csv",header = XGB.classes_ )


output = pd.DataFrame(pred.columns)
output.to_csv("/Users/grazim/Documents/Kaggle_Local/Shelter Animal Outcomes/pred.csv")

test_output = pd.DataFrame(test_pred.columns)
test_output.to_csv("/Users/grazim/Documents/Kaggle_Local/Shelter Animal Outcomes/test_pred.csv")







#COMBINE RF AND XGB PREDICTIONS
XGB_half = test_predictions_XGB/2
RF_half = test_predictions/2

hybrid = XGB_half + RF_half
hybrid.to_csv("/Users/grazim/Documents/Kaggle_Local/Shelter Animal Outcomes/hybrid_pred.csv", header = XGB.classes_ )




