from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import pickle as pickle
def create_model(data):
    # Define X and y
    X = data[['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'goout', 'G1', 'G2', 'absences',
                     'romantic_yes','address_U', 'paid_yes', 'higher_yes', 'internet_yes'  
                     ]]
    y = data['G3']


    # Split the data into train and test set in 80:20 ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # Train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train_scaled , y_train)
   

    # Test the model

    y_pred = model.predict(X_test_scaled)
    print("Mean squared error of the model" , mean_squared_error(y_test , y_pred))

    return model , scaler





def get_data_clean():
    data = pd.read_csv('student-mat.csv' , sep=';')

    data.drop(['Mjob', 'Fjob', 'reason', 'guardian', 'nursery'], axis=1 , inplace=True)
   
    data = pd.get_dummies(data ,  columns=['school', 'sex', 'address', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 'higher', 'internet', 'romantic'], drop_first=True)
    
    le = LabelEncoder()
    data['famsize'] = le.fit_transform(data['famsize'])

    return data

def main():
    
    data = get_data_clean()

    model , scaler = create_model(data)

    with open('model.pkl' , 'wb') as f:
        pickle.dump(model ,f )
     
    with open("scaler.pkl" , 'wb') as f:
        pickle.dump(scaler , f)


if __name__ == '__main__':
    main()