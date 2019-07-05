from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error

    ###Specify the file path to the data###
Titanic_train_file_path = 'C:/Users/Damien/Desktop/Downloads/titanic/train.csv'
Titanic_test_file_path = 'C:/Users/Damien/Desktop/Downloads/titanic/test.csv'
Titanic_train_data = pd.read_csv(Titanic_train_file_path)
Titanic_test_data = pd.read_csv(Titanic_test_file_path)
#All_data = (Titanic_train_data , Titanic_test_data)


#print(Titanic_test_data.PassengerId)


    ###Print data types of list###
#print (Titanic_train_data.columns)
#print (Titanic_train_data.dtypes)

    ###Drop features that won't enrich our data###
Drop_features = ['Name' , 'Ticket' , 'Cabin' ,
'Embarked' , 'SibSp' , 'Parch'] 
Titanic_train_data = Titanic_train_data.drop(Drop_features, axis=1)
Titanic_test_data = Titanic_test_data.drop(Drop_features, axis=1)
    
    ###Fill missing data to make more meaningful results###
Titanic_train_data['Age'] = Titanic_train_data['Age'].fillna(Titanic_train_data['Age'].mean())
Titanic_test_data['Age'] = Titanic_test_data['Age'].fillna(Titanic_test_data['Age'].mean())
Titanic_test_data['Fare'] = Titanic_test_data['Fare'].fillna(Titanic_test_data['Fare'].mean())

    ###Convert all objects to int###
Titanic_train_data['Age'] = Titanic_train_data['Age'].astype(int)
Titanic_test_data['Age'] = Titanic_test_data['Age'].astype(int)
    ###Cut data to bins###
#Titanic_train_data['Fare'] = pd.qcut(Titanic_train_data['Fare'],4)
#Titanic_train_data['Age'] = pd.qcut(Titanic_train_data['Age'],6)
 
    ####Look at data and understand what it implies#####
#print (Titanic_train_data[['Pclass' , 'Survived']].groupby(['Pclass'], ).mean())
#print (Titanic_train_data[['Sex' , 'Survived']].groupby(['Sex']).mean())
#print (Titanic_train_data[['Fare' , 'Survived']].groupby(['Fare']).mean())
#print (Titanic_train_data[['Age' , 'Survived']].groupby(['Age']).mean())

     ###Print NANs in data then we can fill in to make more meaningful data###
#print (Titanic_test_data.columns)
#print (Titanic_test_data['Age'].isna().sum())
#print (Titanic_test_data['Fare'].isna().sum())
#print (Titanic_train_data['Age'].isna().sum())
#print (Titanic_train_data['Fare'].isna().sum())
#print (All_data)
    ###Mapping the data===Not all the data is in numerical values so we must
    ###convert all data to this type###
    
All_data = (Titanic_train_data , Titanic_test_data)    

for data in All_data:
    
    
    ###Mapping sex###
    sex_map = {'female':0 , 'male':1}
    data['Sex'] = data['Sex'].map(sex_map).astype(int)
    
        ###Mapping the fare###
    data.loc[ data['Fare']<= 7.91, 'Fare'] = 0   
    data.loc[ (data['Fare']> 7.91) & (data['Fare']<= 14.454) , 'Fare'] = 1
    data.loc[ (data['Fare']> 14.454) & (data['Fare']<= 31.0)  , 'Fare'] = 2
    data.loc[ data['Fare']> 31.0 , 'Fare'] = 3
    data['Fare'] = data['Fare'].astype(int)
    
        ###Mapping ages of people###
    data.loc[ data['Age']<=19.0 ,'Age']=0
    data.loc[ (data['Age']> 19.0) & (data['Age']<= 25.0) , 'Age'] = 1
    data.loc[ (data['Age']> 25.0) & (data['Age']<= 29.699) , 'Age'] = 2
    data.loc[ (data['Age']> 29.699) & (data['Age']<= 31.0) , 'Age'] = 3
    data.loc[ (data['Age']> 31.0) & (data['Age']<= 40.5) , 'Age'] = 4
    data.loc[ data['Age']> 40.5 , 'Age'] = 5
    
    

#print(Titanic_train_data.dtypes)
#print(Titanic_test_data.dtypes)
    ###Train the model###
X_train = Titanic_train_data.drop(['Survived' , 'PassengerId'], axis=1)
Y_train = Titanic_train_data.Survived
X_test = Titanic_test_data.drop('PassengerId' , axis=1)
Titanic_model = DecisionTreeClassifier(random_state=1)
Titanic_model.fit(X_train, Y_train)
Y = Titanic_model.predict(X_test)
accuracy = round(Titanic_model.score(X_train, Y_train),2)

#print (Y)
print (accuracy)

    ###Output to csv file for submission###

Titanic_submission = pd.DataFrame({'PassengerId': Titanic_test_data['PassengerId'],
'Survived' : Y})
Titanic_submission.to_csv('Titanic_submission.csv' , index=False)
                       ###plot data for better representation###
#sns.countplot(Titanic_train_data['Pclass'], hue=Titanic_train_data['Sex'])
plt.show    
