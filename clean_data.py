import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

def process_data():
    print("🚀 Starting Data Pipeline...")
    
    # 1. Load Data
    df = pd.read_csv('youtube_ad_revenue_dataset.csv')
    
    # 2. Preprocessing
    df.drop_duplicates(inplace=True)
    
    # Handle Missing Values (Numeric: Median, Categorical: Mode)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = ['category', 'device', 'country']
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    # 3. Feature Engineering
    df['engagement_rate'] = (df['likes'] + df['comments']) / (df['views'] + 1)
    
    # 4. Encoding (Crucial for the Streamlit App later)
    # We save the encoder so the App knows 'Gaming' = 1, 'Music' = 2, etc.
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    # 5. Save Everything
    df.to_csv('cleaned_data.csv', index=False)
    joblib.dump(encoders, 'label_encoders.pkl')
    
    print("✅ Success!")
    print(f"Final Row Count: {len(df)}")
    print("Files Created: 'cleaned_data.csv' and 'label_encoders.pkl'")

if __name__ == "__main__":
    process_data()
