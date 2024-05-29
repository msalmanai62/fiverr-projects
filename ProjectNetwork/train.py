import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def preprocesss(train_data):
            # start_time = time.time()
            Y_data = train_data['class']
            X_data = train_data.drop(['class'], axis=1)
            categorical_cols = ['protocol_type', 'service', 'flag']
            encoded_cols = pd.get_dummies(X_data[categorical_cols])
            # Step 3: Concatenate original dataset with encoded columns
            X_data_encoded = pd.concat([X_data, encoded_cols], axis=1)

            X_data_encoded.drop(categorical_cols, axis=1, inplace=True)

            scaler = StandardScaler()
            X_data_scaled = scaler.fit_transform(X_data_encoded)
            X_train, X_test, y_train, y_test = train_test_split(X_data_scaled, Y_data, test_size=0.2, random_state=30)

        #     rfe = RFE(RandomForestClassifier(), n_features_to_select=10)
        #     X_train = rfe.fit_transform(X_train, y_train)

        #     X_test = rfe.transform(X_test)

            data = {
                    'X_train':X_train,
                    'y_train':y_train,
                    'X_test':X_test,
                    'y_test':y_test,
            }
            print("Preprocessing Done")
            return data

def trainn(modell, data):
            start_time = time.time()
            modell.fit(data['X_train'], data['y_train'])
            print('training done')
            end_time = time.time()
            training_time = end_time - start_time

            print("Prediction Done!")
            predictions = modell.predict(data['X_test'])

            accuracy = accuracy_score(data['y_test'], predictions)
            return accuracy, training_time