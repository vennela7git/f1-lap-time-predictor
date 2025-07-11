from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load the data
def load_lap_data(csv_path = 'lap_data_clean.csv'):
    return pd.read_csv(csv_path)

#seperate features and target
def train_model(df):
    X = df[['Driver','Compound','Stint','AirTemp','TrackTemp','Rainfall']]
    y= df['LapTime']

# categorical columns to encode
    cat_cols = ['Driver','Compound']

# define preprocessing    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), cat_cols)
        ],
        remainder='passthrough'
    )
#define model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regression', RandomForestRegressor(n_estimators = 100, random_state = 42))
     ])    
# train, test and split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#fit the model
    model.fit(X_train, y_train)

    return model, X_test, y_test
