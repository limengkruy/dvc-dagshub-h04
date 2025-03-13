import pandas as pd
from sklearn.model_selection import train_test_split

def clean_data(input_file, output_file):
    df = pd.read_csv(input_file)
    
    # Handle missing values in TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Drop unnecessary columns
    df.drop(columns=['customerID'], inplace=True)

    df.to_csv(output_file, index=False)

def encode_features(input_file, output_file):
    df = pd.read_csv(input_file)

    # One-hot encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    df.to_csv(output_file, index=False)

def split_data(input_file, train_file, test_file):
    df = pd.read_csv(input_file)

    X = df.drop(columns=['Churn_Yes'])
    y = df['Churn_Yes']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pd.concat([X_train, y_train], axis=1).to_csv(train_file, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_file, index=False)

if __name__ == "__main__":
    raw_data = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    cleaned_data = 'data/cleaned_data.csv'
    encoded_data = 'data/encoded_data.csv'
    train_data = 'data/train_data.csv'
    test_data = 'data/test_data.csv'

    clean_data(raw_data, cleaned_data)
    encode_features(cleaned_data, encoded_data)
    split_data(encoded_data, train_data, test_data)