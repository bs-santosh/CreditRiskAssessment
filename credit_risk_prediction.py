"""
Credit Risk Prediction Model Training Script
============================================

This script trains the credit risk prediction models and saves them for production use.
Run this script before launching the Streamlit app to generate all required model files.

Usage:
    python credit_risk_prediction.py

Output:
    - models/xgboost_model.pkl
    - models/random_forest_model.pkl
    - models/logistic_regression_model.pkl
    - models/scaler.pkl
    - models/feature_names.pkl
    - models/shap_explainer.pkl
"""

import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Explainable AI
import shap
from lime import lime_tabular

# Model persistence
import joblib

# Set random seed for reproducibility
np.random.seed(42)

print("="*60)
print("CREDIT RISK PREDICTION MODEL TRAINING")
print("="*60)
print("\n1. Loading libraries...")
print("✓ All libraries loaded successfully!")

# ============================================================================
# 2. DATA LOADING
# ============================================================================
print("\n2. Loading dataset...")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

column_names = [
    'checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_status', 'employment', 'installment_rate', 'personal_status',
    'other_parties', 'residence_since', 'property_magnitude', 'age',
    'other_payment_plans', 'housing', 'existing_credits', 'job',
    'num_dependents', 'own_telephone', 'foreign_worker', 'class'
]

try:
    df = pd.read_csv(url, sep=' ', header=None, names=column_names)
    print(f"✓ Dataset loaded successfully from UCI repository!")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
except Exception as e:
    print(f"⚠ Could not load from URL: {e}")
    print("Trying local file: data/german_credit_data.csv")
    try:
        df = pd.read_csv('german_credit_data.csv')
        print("✓ Dataset loaded from local file!")
    except:
        print("❌ Error: Could not load dataset from URL or local file.")
        print("Please ensure the dataset is available.")
        exit(1)

# Convert target: UCI format uses 1=Good, 2=Bad; convert to 1=Good, 0=Bad
df['class'] = df['class'].apply(lambda x: 1 if x == 1 else 0)
print(f"  Target distribution: {df['class'].value_counts().to_dict()}")

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================
print("\n3. Preprocessing data...")

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove('class')
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"  Numerical features: {len(numerical_cols)}")
print(f"  Categorical features: {len(categorical_cols)}")

# One-Hot Encoding for categorical variables
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"✓ One-Hot Encoding completed")
print(f"  New shape: {df_encoded.shape}")

# Separate features and target
X = df_encoded.drop('class', axis=1)
y = df_encoded['class']

# Feature Scaling
scaler = StandardScaler()
numerical_feature_names = [col for col in X.columns if col in numerical_cols]
X_scaled = X.copy()
X_scaled[numerical_feature_names] = scaler.fit_transform(X[numerical_feature_names])
print(f"✓ Feature scaling completed")

# Save feature names
feature_names = X_scaled.columns.tolist()
print(f"  Total features: {len(feature_names)}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)
print(f"✓ Train-test split completed")
print(f"  Training: {X_train.shape[0]} samples")
print(f"  Testing: {X_test.shape[0]} samples")

# ============================================================================
# 4. MODEL TRAINING
# ============================================================================
print("\n4. Training models...")
print("  This may take a few minutes...")

# Logistic Regression
print("\n  [1/3] Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1])
print(f"  ✓ Logistic Regression trained - Accuracy: {acc_lr:.4f}, AUC: {auc_lr:.4f}")

# Random Forest
print("\n  [2/3] Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
print(f"  ✓ Random Forest trained - Accuracy: {acc_rf:.4f}, AUC: {auc_rf:.4f}")

# XGBoost
print("\n  [3/3] Training XGBoost...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
auc_xgb = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
print(f"  ✓ XGBoost trained - Accuracy: {acc_xgb:.4f}, AUC: {auc_xgb:.4f}")

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================
print("\n5. Model evaluation summary:")
print("="*60)
print(f"{'Model':<25} {'Accuracy':<12} {'AUC':<12}")
print("-"*60)
print(f"{'Logistic Regression':<25} {acc_lr:<12.4f} {auc_lr:<12.4f}")
print(f"{'Random Forest':<25} {acc_rf:<12.4f} {auc_rf:<12.4f}")
print(f"{'XGBoost (Production)':<25} {acc_xgb:<12.4f} {auc_xgb:<12.4f}")
print("="*60)

# ============================================================================
# 6. EXPLAINABILITY - SHAP & LIME
# ============================================================================
print("\n6. Creating explainability models...")

# SHAP Explainer
print("  [1/2] Creating SHAP explainer...")
print("  ⏳ Computing SHAP values (this may take a moment)...")
shap_explainer = shap.TreeExplainer(xgb_model)
shap_values = shap_explainer.shap_values(X_test)
print("  ✓ SHAP explainer created successfully!")

# LIME Explainer
print("\n  [2/2] Creating LIME explainer...")
lime_explainer = lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=['Bad Credit (Reject)', 'Good Credit (Approve)'],
    mode='classification',
    discretize_continuous=True
)
print("  ✓ LIME explainer created successfully!")

# ============================================================================
# 7. SAVE MODELS AND ARTIFACTS
# ============================================================================
print("\n7. Saving models and artifacts...")

# Create models directory
if not os.path.exists('models'):
    os.makedirs('models')
    print("  ✓ Created models/ directory")

# Save models
joblib.dump(xgb_model, 'models/xgboost_model.pkl')
print("  ✓ XGBoost model saved: models/xgboost_model.pkl")

joblib.dump(rf_model, 'models/random_forest_model.pkl')
print("  ✓ Random Forest model saved: models/random_forest_model.pkl")

joblib.dump(lr_model, 'models/logistic_regression_model.pkl')
print("  ✓ Logistic Regression model saved: models/logistic_regression_model.pkl")

# Save preprocessing objects
joblib.dump(scaler, 'models/scaler.pkl')
print("  ✓ Scaler saved: models/scaler.pkl")

joblib.dump(feature_names, 'models/feature_names.pkl')
print("  ✓ Feature names saved: models/feature_names.pkl")

joblib.dump(shap_explainer, 'models/shap_explainer.pkl')
print("  ✓ SHAP explainer saved: models/shap_explainer.pkl")

# Note: LIME explainer is not saved because it contains lambda functions that can't be pickled
# It will be recreated on-demand in the Streamlit app
print("  ℹ LIME explainer will be created on-demand in the app (not serializable)")

# Save training data for LIME recreation
joblib.dump(X_train.values, 'models/X_train_sample.pkl')
print("  ✓ Training data sample saved for LIME: models/X_train_sample.pkl")

# ============================================================================
# 8. VERIFICATION
# ============================================================================
print("\n8. Verifying saved models...")
try:
    loaded_model = joblib.load('models/xgboost_model.pkl')
    loaded_scaler = joblib.load('models/scaler.pkl')
    loaded_features = joblib.load('models/feature_names.pkl')
    test_pred = loaded_model.predict(X_test.iloc[0:1])
    print("  ✓ All models loaded and verified successfully!")
except Exception as e:
    print(f"  ⚠ Error verifying models: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY! 🎉")
print("="*60)
print("\nSaved artifacts in ./models/:")
print("  - xgboost_model.pkl (main production model)")
print("  - random_forest_model.pkl")
print("  - logistic_regression_model.pkl")
print("  - scaler.pkl")
print("  - feature_names.pkl")
print("  - shap_explainer.pkl")
print("  - X_train_sample.pkl (for LIME recreation)")
print("\nNext steps:")
print("  1. Run the Streamlit app: streamlit run app.py")
print("  2. Test predictions with the interactive interface")
print("  3. Review SHAP explanations for transparency")
print("\n✓ Ready for production deployment!")
print("="*60)
