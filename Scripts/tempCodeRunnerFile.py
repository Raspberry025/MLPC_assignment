import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestClassifier, 
                              HistGradientBoostingRegressor)
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, roc_auc_score, mean_squared_error, r2_score, f1_score)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load the dataset
df = pd.read_csv('D:\\Flood_Detection\\Dataset\\flood.csv\\flood.csv')
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# Data Cleaning
df = df.dropna()
df = df.drop_duplicates()

# Feature Engineering
df['Monsoon_Drainage'] = df['MonsoonIntensity'] * df['DrainageSystems']
df['Urban_Watershed'] = df['Urbanization'] * df['Watersheds']

features = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
    'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
    'Siltation', 'AgriculturalPractices', 'Encroachments',
    'IneffectiveDisasterPreparedness', 'DrainageSystems',
    'CoastalVulnerability', 'Landslides', 'Watersheds',
    'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
    'InadequatePlanning', 'PoliticalFactors', 'Monsoon_Drainage', 'Urban_Watershed'
]
target = 'FloodProbability'

X = df[features]
y = df[target]

# Train-Validation-Test Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Model Training (Regression)
model_reg = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=8,
    min_samples_leaf=10,
    min_samples_split=5,
    random_state=42
)
model_reg.fit(X_train_scaled, y_train)

print("\n--- Training HistGradientBoostingRegressor (single-core) ---")
start_time = time.time()
hgb_reg_single = HistGradientBoostingRegressor(
    max_iter=200,
    max_depth=8,
    min_samples_leaf=10,
    random_state=42,
    n_iter_no_change=10,
    early_stopping=True,
    verbose=0
)
hgb_reg_single.fit(X_train_scaled, y_train)
single_time = time.time() - start_time
print(f"Single-core training time (regression): {single_time:.2f} seconds")

print("\n--- Training HistGradientBoostingRegressor (multi-core) ---")
start_time = time.time()
hgb_reg_multi = HistGradientBoostingRegressor(
    max_iter=200,
    max_depth=8,
    min_samples_leaf=10,
    random_state=42,
    n_iter_no_change=10,
    early_stopping=True,
    verbose=0
)
hgb_reg_multi.fit(X_train_scaled, y_train)
multi_time = time.time() - start_time
print(f"Multi-core training time (regression): {multi_time:.2f} seconds")

def evaluate_regression(model, X, y, set_name):
    y_pred = model.predict(X)
    print(f"\n{set_name} Regression Evaluation:")
    print(f"R2 Score: {r2_score(y, y_pred):.2f}")
    print(f"RMSE: {mean_squared_error(y, y_pred, squared=False):.2f}")
    plt.scatter(y, y_pred, alpha=0.3)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{set_name} Actual vs Predicted (Regression)")
    plt.show()

print("\n--- HistGradientBoostingRegressor Performance (Single-core) ---")
evaluate_regression(hgb_reg_single, X_test_scaled, y_test, "Test (Single-core)")

print("\n--- HistGradientBoostingRegressor Performance (Multi-core) ---")
evaluate_regression(hgb_reg_multi, X_test_scaled, y_test, "Test (Multi-core)")

# Feature Importance
importances = model_reg.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()

# Prediction and Warning System
def flood_warning_system(input_data):
    input_df = pd.DataFrame([input_data], columns=features)
    scaled_data = scaler.transform(input_df)
    proba = model_reg.predict(scaled_data)[0]
    thresholds = {
        'green': 0.4,
        'yellow': 0.65,
        'orange': 0.85,
        'red': 1.0
    }
    if proba < thresholds['green']:
        level = "Normal"
        color = "green"
        action = "No action needed"
    elif proba < thresholds['yellow']:
        level = "Watch"
        color = "yellow"
        action = "Monitor weather updates"
    elif proba < thresholds['orange']:
        level = "Warning"
        color = "orange"
        action = "Prepare emergency supplies"
    else:
        level = "Emergency"
        color = "red"
        action = "Evacuate if in flood-prone area"
    explanation = []
    for i, feature in enumerate(features):
        value = input_data[i]
        if value > df[feature].quantile(0.9):
            explanation.append(f"High {feature} ({value})")
    return {
        'probability': float(proba),
        'warning_level': level,
        'color_code': color,
        'recommended_action': action,
        'key_factors': explanation,
        'threshold_breaches': [f for f in features 
                              if (input_data[features.index(f)] > df[f].quantile(0.9))]
    }

# Test Cases (each must have 22 values)
test_cases = [
    [2, 5, 4, 1, 3, 2, 4, 3, 2, 1, 3, 5, 1, 1, 4, 3, 2, 1, 2, 3, 6, 3],
    [6, 7, 6, 5, 6, 5, 7, 4, 5, 4, 7, 2, 4, 3, 6, 5, 6, 4, 5, 6, 12, 18],
    [9, 9, 8, 7, 8, 7, 9, 1, 7, 6, 9, 1, 7, 6, 8, 7, 8, 6, 7, 8, 72, 48]
]

for case in test_cases:
    result = flood_warning_system(case)
    print("\n" + "="*50)
    print(f"Input: {case}")
    print(f"Flood Probability: {result['probability']:.1%}")
    print(f"Warning Level: {result['warning_level']} ({result['color_code']})")
    print(f"Key Factors: {', '.join(result['key_factors'])}")
    print(f"Action: {result['recommended_action']}")

# Example test case
test_case = [2, 5, 4, 1, 3, 2, 4, 3, 2, 1, 3, 5, 1, 1, 4, 3, 2, 1, 2, 3, 6, 3]
result = flood_warning_system(test_case)
print(result)

# Save the model pipeline
pipeline = {
    'model': model_reg,
    'scaler': scaler,
    'features': features,
    'thresholds': {
        'watch': 0.3,
        'warning': 0.6,
        'emergency': 0.8
    }
}
joblib.dump(pipeline, 'flood_warning_pipeline.pkl')

# --- CLASSIFICATION MODEL COMPARISON (RandomForestClassifier only) ---
df['FloodClass'] = (df['FloodProbability'] >= 0.5).astype(int)
target_clf = 'FloodClass'
X_clf = df[features]
y_clf = df[target_clf]

X_train_clf, X_temp_clf, y_train_clf, y_temp_clf = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42, stratify=y_clf)
X_val_clf, X_test_clf, y_val_clf, y_test_clf = train_test_split(
    X_temp_clf, y_temp_clf, test_size=0.5, random_state=42, stratify=y_temp_clf)

scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_val_clf_scaled = scaler_clf.transform(X_val_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

# Single-core
print("\n--- Training RandomForestClassifier (single-core) ---")
start_time = time.time()
model_clf_single = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    min_samples_split=5,
    random_state=42,
    n_jobs=1
)
model_clf_single.fit(X_train_clf_scaled, y_train_clf)
clf_single_time = time.time() - start_time
print(f"Single-core training time (classification): {clf_single_time:.2f} seconds")

# Multi-core
print("\n--- Training RandomForestClassifier (multi-core, parallel) ---")
start_time = time.time()
model_clf_parallel = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
model_clf_parallel.fit(X_train_clf_scaled, y_train_clf)
clf_parallel_time = time.time() - start_time
print(f"Multi-core training time (classification): {clf_parallel_time:.2f} seconds")

def evaluate_classification(model, X, y, set_name):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    print(f"\n{set_name} Classification Evaluation:")
    print(f"Accuracy: {accuracy_score(y, y_pred):.2f}")
    print(f"F1 Score: {f1_score(y, y_pred):.2f}")
    print(f"ROC AUC: {roc_auc_score(y, y_proba):.2f}")
    print("Classification Report:\n", classification_report(y, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))

print("\n--- Classification Model Performance (Single-core) ---")
evaluate_classification(model_clf_single, X_test_clf_scaled, y_test_clf, "Test (Single-core)")

print("\n--- Classification Model Performance (Multi-core) ---")
evaluate_classification(model_clf_parallel, X_test_clf_scaled, y_test_clf, "Test (Multi-core)")

# Confusion Matrix Visualization
def plot_confusion_matrix(model, X, y, set_name):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Flood', 'Flood'],
                yticklabels=['No Flood', 'Flood'])
    plt.title(f'{set_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

plot_confusion_matrix(model_clf_parallel, X_train_clf_scaled, y_train_clf, "Training")
plot_confusion_matrix(model_clf_parallel, X_val_clf_scaled, y_val_clf, "Validation")
plot_confusion_matrix(model_clf_parallel, X_test_clf_scaled, y_test_clf, "Test")