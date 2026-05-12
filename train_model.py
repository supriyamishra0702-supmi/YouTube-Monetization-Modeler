import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# 1. Load the cleaned data
print("📂 Loading data...")
df = pd.read_csv('cleaned_data.csv')

# 2. Define Features and Target (Exact order for Streamlit)
feature_names = [
    'views', 'likes', 'comments', 'watch_time_minutes', 
    'video_length_minutes', 'subscribers', 'category', 
    'device', 'country', 'engagement_rate'
]

X = df[feature_names] 
y = df['ad_revenue_usd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define 5 Models (Following Project Approach)
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(alpha=0.1),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

# 4. Train and Compare 
print("\n📊 MODEL SHOOTOUT RESULTS")
print(f"{'Model Name':<20} | {'R2':<7} | {'MAE':<8} | {'RMSE':<8}")
print("-" * 55)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    # Calculate RMSE (Square Root of Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    print(f"{name:<20} | {r2:<7.4f} | ${mae:<7.2f} | ${rmse:<7.2f}")

# 5. SAVE THE WINNER
# We save RandomForest because it has the highest R2 and lowest MAE/RMSE
joblib.dump(models["Random Forest"], 'final_model.pkl')

print("-" * 55)
print(f"🏆 SUCCESS: 'final_model.pkl' (Random Forest) is ready for Streamlit.")
print(f"\n🏆 Model Saved: {final_model_name}")
print("✅ SUCCESS: 'final_model.pkl' is now ready for your interactive Streamlit app.")
