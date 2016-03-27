import pandas as pd


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
    #data['age_years'] = pd.to_numeric(data['age_years'])
    
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
    
    data = data.drop(['DateTime','SexuponOutcome','AgeuponOutcome','breed_category','Name'], axis = 1)
    
    #Drop unneeded columns
    data = data.drop(['DateTime','SexuponOutcome','AgeuponOutcome'], axis = 1)
    
    
    return(data)
    
    
train_import = pd.read_csv("/Users/grazim/Documents/Kaggle_Local/Shelter Animal Outcomes/train.csv")

    
    
    
train = data_import("/Users/grazim/Documents/Kaggle_Local/Shelter Animal Outcomes/train.csv")
