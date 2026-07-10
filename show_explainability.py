"""
Display Explainability Results from Credit Risk Models
"""
import pandas as pd
import numpy as np
import joblib
import shap
from lime import lime_tabular

print("="*70)
print("CREDIT RISK PREDICTION - EXPLAINABILITY RESULTS")
print("="*70)

# Load models and data
print("\n1. Loading models and test data...")
model = joblib.load('models/xgboost_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')
shap_explainer = joblib.load('models/shap_explainer.pkl')
X_train_sample = joblib.load('models/X_train_sample.pkl')

print(f"   ✓ Model loaded: XGBoost")
print(f"   ✓ Features: {len(feature_names)}")

# Create LIME explainer
lime_explainer = lime_tabular.LimeTabularExplainer(
    X_train_sample,
    feature_names=feature_names,
    class_names=['Bad Credit (Reject)', 'Good Credit (Approve)'],
    mode='classification',
    discretize_continuous=True
)
print(f"   ✓ Explainers ready: SHAP & LIME")

# Load sample test data
print("\n2. Loading sample prediction...")
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
except:
    df = pd.read_csv('data/german_credit_data.csv')

df['class'] = df['class'].apply(lambda x: 1 if x == 1 else 0)

# Process one sample
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove('class')
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
X = df_encoded.drop('class', axis=1)
y = df_encoded['class']

# Scale numerical features
X_scaled = X.copy()
numerical_feature_names = [col for col in X.columns if col in numerical_cols]
X_scaled[numerical_feature_names] = scaler.transform(X[numerical_feature_names])

# Ensure correct feature order
for col in feature_names:
    if col not in X_scaled.columns:
        X_scaled[col] = 0
X_scaled = X_scaled[feature_names]

# Take first test sample
sample_idx = 5
test_sample = X_scaled.iloc[sample_idx:sample_idx+1]
true_label = y.iloc[sample_idx]

print(f"   ✓ Sample #{sample_idx} selected")
print(f"   True Label: {'Good Credit' if true_label == 1 else 'Bad Credit'}")

# Make prediction
prediction = model.predict(test_sample)[0]
probabilities = model.predict_proba(test_sample)[0]

print(f"   Prediction: {'Good Credit' if prediction == 1 else 'Bad Credit'}")
print(f"   Confidence: {probabilities[prediction]*100:.2f}%")

# ============================================================================
# SHAP EXPLAINABILITY
# ============================================================================
print("\n" + "="*70)
print("SHAP EXPLAINABILITY RESULTS")
print("="*70)

shap_values = shap_explainer.shap_values(test_sample)

print("\nTop 10 Features by SHAP Impact:")
print("-"*70)
shap_df = pd.DataFrame({
    'Feature': feature_names,
    'SHAP Value': shap_values[0],
    'Feature Value': test_sample.values[0]
})
shap_df['Abs_SHAP'] = np.abs(shap_df['SHAP Value'])
shap_df_sorted = shap_df.sort_values('Abs_SHAP', ascending=False).head(10)

for idx, row in shap_df_sorted.iterrows():
    impact = "↑ Increases Good Credit" if row['SHAP Value'] > 0 else "↓ Increases Bad Credit"
    print(f"{row['Feature']:<35} {row['SHAP Value']:>8.4f}  {impact}")

print(f"\nBase Value (Average): {shap_explainer.expected_value:.4f}")
print(f"SHAP Sum: {shap_values[0].sum():.4f}")
print(f"Final Prediction Score: {shap_explainer.expected_value + shap_values[0].sum():.4f}")

# ============================================================================
# LIME EXPLAINABILITY
# ============================================================================
print("\n" + "="*70)
print("LIME EXPLAINABILITY RESULTS")
print("="*70)

lime_exp = lime_explainer.explain_instance(
    test_sample.values[0],
    model.predict_proba,
    num_features=10,
    top_labels=2
)

print("\nPrediction Probabilities:")
print("-"*70)
print(f"Bad Credit (Reject):  {lime_exp.predict_proba[0]*100:.2f}%")
print(f"Good Credit (Approve): {lime_exp.predict_proba[1]*100:.2f}%")

print(f"\nTop 10 Features Influencing the Prediction (Class {prediction}):")
print("-"*70)
exp_list = lime_exp.as_list(label=prediction)
for feature_cond, weight in exp_list:
    impact = "Supports" if weight > 0 else "Opposes"
    print(f"{feature_cond:<45} {weight:>8.4f}  {impact}")

print(f"\nIntercept (Base): {lime_exp.intercept[prediction]:.4f}")
print(f"Total Weight: {sum([w for _, w in exp_list]):.4f}")

# ============================================================================
# FEATURE IMPORTANCE (Global)
# ============================================================================
print("\n" + "="*70)
print("GLOBAL FEATURE IMPORTANCE (XGBoost)")
print("="*70)

feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False).head(15)

print("\nTop 15 Most Important Features:")
print("-"*70)
for idx, row in importance_df.iterrows():
    bar_length = int(row['Importance'] * 50)
    bar = '█' * bar_length
    print(f"{row['Feature']:<35} {row['Importance']:>6.4f}  {bar}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("EXPLAINABILITY SUMMARY")
print("="*70)
print("\n✓ SHAP Analysis:")
print(f"  - Base prediction value: {shap_explainer.expected_value:.4f}")
print(f"  - Top positive feature: {shap_df_sorted[shap_df_sorted['SHAP Value'] > 0].iloc[0]['Feature'] if len(shap_df_sorted[shap_df_sorted['SHAP Value'] > 0]) > 0 else 'None'}")
print(f"  - Top negative feature: {shap_df_sorted[shap_df_sorted['SHAP Value'] < 0].iloc[0]['Feature'] if len(shap_df_sorted[shap_df_sorted['SHAP Value'] < 0]) > 0 else 'None'}")

print("\n✓ LIME Analysis:")
print(f"  - Model confidence: {max(lime_exp.predict_proba)*100:.2f}%")
print(f"  - Prediction class: {'Good Credit' if prediction == 1 else 'Bad Credit'}")
print(f"  - Number of features analyzed: {len(exp_list)}")

print("\n✓ Model Performance:")
print(f"  - Model: XGBoost Classifier")
print(f"  - Features used: {len(feature_names)}")
print(f"  - Explainability methods: SHAP (global + local), LIME (local)")

print("\n" + "="*70)
print("For visual explanations, run: streamlit run app.py")
print("="*70)
