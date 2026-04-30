import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# 1. Load the cleaned data
print("📂 Loading data...")
df = pd.read_csv('cleaned_data.csv')

# 2. Define Features and Target
# We define the EXACT order of columns to ensure the app matches the model
feature_names = [
    'views', 'likes', 'comments', 'watch_time_minutes', 
    'video_length_minutes', 'subscribers', 'category', 
    'device', 'country', 'engagement_rate'
]

X = df[feature_names] 
y = df['ad_revenue_usd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define 5 Models
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(),
    "DecisionTree": DecisionTreeRegressor(max_depth=None),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=None, n_jobs=-1, random_state=42),
    "AdaBoost": AdaBoostRegressor()
}

# 4. Train and Compare
print("\n🤖 Training Models (This may take a minute)...")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"{name:12} | R2: {r2:.4f} | MAE: ${mae:.2f}")

# 5. FORCED SAVE: Random Forest
# We save RandomForest because it is much more sensitive to Country/Category changes
final_model_name = "RandomForest"
joblib.dump(models[final_model_name], 'final_model.pkl')

print(f"\n🏆 Model Saved: {final_model_name}")
print("✅ SUCCESS: 'final_model.pkl' is now ready for your interactive Streamlit app.")
