import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


# 1. GENERATE SYNTHETIC BEHAVIORAL DATA (WITH NOISE)
print("Generating synthetic behavioral dataset...")
np.random.seed(42)
n_samples = 5000

sleep_hours = np.random.normal(7, 1.5, n_samples)
screen_time = np.random.normal(4, 2, n_samples) 
focus_hours = np.random.normal(3, 1.5, n_samples)
water_intake = np.random.normal(2, 1, n_samples) 

# The base logic
raw_score = (
    (focus_hours * 0.4) + 
    (sleep_hours * 0.2) - 
    (screen_time * 0.3) + 
    (water_intake * 0.1)
)

# THE FIX: Add statistical noise (luck/external factors)
noise = np.random.normal(0, 0.8, n_samples) 
noisy_score = raw_score + noise

# Convert to a probability curve (Sigmoid function)
probability_curve = 1 / (1 + np.exp(-noisy_score))

# Flip a weighted coin based on that probability to get 1 or 0
goal_achieved = np.random.binomial(1, probability_curve)

df = pd.DataFrame({
    'sleep_hours': np.clip(sleep_hours, 2, 12),
    'screen_time': np.clip(screen_time, 0, 12),
    'focus_hours': np.clip(focus_hours, 0, 10),
    'water_intake': np.clip(water_intake, 0, 5),
    'goal_achieved': goal_achieved
})

# 2. TRAIN THE XGBOOST MODEL
print("Training XGBoost Predictive Engine...")
X = df.drop('goal_achieved', axis=1)
y = df['goal_achieved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# 3. SAVE THE MODEL
joblib.dump(model, 'behavioral_model.pkl')
print("Model saved as 'behavioral_model.pkl'. Ready for deployment!")