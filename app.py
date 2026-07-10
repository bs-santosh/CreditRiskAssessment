"""
Explainable Credit Risk Prediction Application
==============================================

A production-ready Streamlit application for credit risk assessment with explainable AI.

This application provides:
- Interactive credit risk predictions
- SHAP-based explanations for transparency
- LIME explanations for local interpretability
- Professional bank-like interface
- Real-time feature impact visualization

Author: Santosh Balgar Sachchidananda
Date: June 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime import lime_tabular
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .subheader {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 20px;
        margin-bottom: 15px;
        border-bottom: 2px solid #3498db;
        padding-bottom: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

@st.cache_resource
def load_models():
    """Load trained models and preprocessing objects"""
    try:
        model = joblib.load('models/xgboost_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        shap_explainer = joblib.load('models/shap_explainer.pkl')
        X_train_sample = joblib.load('models/X_train_sample.pkl')
        return model, scaler, feature_names, shap_explainer, X_train_sample, True
    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}")
        st.info("Please run the training script first: python credit_risk_prediction.py")
        return None, None, None, None, None, False

@st.cache_resource
def create_lime_explainer(_X_train_sample, feature_names):
    """Create LIME explainer on-demand (can't be pickled)"""
    return lime_tabular.LimeTabularExplainer(
        _X_train_sample,
        feature_names=feature_names,
        class_names=['Bad Credit (Reject)', 'Good Credit (Approve)'],
        mode='classification',
        discretize_continuous=True
    )

def create_input_features():
    """Create sidebar input form for credit application"""
    st.sidebar.markdown("<h2 style='color: #1f77b4;'>📋 Credit Application Form</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    # Personal Information
    st.sidebar.markdown("### 👤 Personal Information")
    age = st.sidebar.slider("Age", 18, 75, 35, help="Applicant's age in years")
    
    # Financial Information
    st.sidebar.markdown("### 💰 Financial Information")
    credit_amount = st.sidebar.number_input(
        "Credit Amount (€)", 
        min_value=250, 
        max_value=20000, 
        value=3000, 
        step=100,
        help="Requested loan amount in Euros"
    )
    
    duration = st.sidebar.slider(
        "Loan Duration (months)", 
        4, 72, 24,
        help="Loan repayment period in months"
    )
    
    installment_rate = st.sidebar.slider(
        "Installment Rate (%)", 
        1, 4, 2,
        help="Installment as percentage of disposable income"
    )
    
    # Account Status
    st.sidebar.markdown("### 🏦 Account Status")
    checking_status = st.sidebar.selectbox(
        "Checking Account Status",
        ["<0", "0<=X<200", ">=200", "no checking"],
        help="Current balance in checking account"
    )
    
    savings_status = st.sidebar.selectbox(
        "Savings Account Status",
        ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"],
        help="Current savings amount"
    )
    
    # Credit History
    st.sidebar.markdown("### 📊 Credit History")
    credit_history = st.sidebar.selectbox(
        "Credit History",
        ["no credits/all paid", "all paid", "existing paid", "delayed previously", "critical/other existing credit"],
        help="Past credit payment behavior"
    )
    
    # Employment
    st.sidebar.markdown("### 💼 Employment")
    employment = st.sidebar.selectbox(
        "Employment Duration",
        ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"],
        help="Years at current employment"
    )
    
    job = st.sidebar.selectbox(
        "Job Type",
        ["unemp/unskilled non res", "unskilled resident", "skilled", "high qualif/self emp/mgmt"],
        help="Employment category"
    )
    
    # Purpose
    st.sidebar.markdown("### 🎯 Loan Purpose")
    purpose = st.sidebar.selectbox(
        "Purpose",
        ["new car", "used car", "furniture/equipment", "radio/tv", "domestic appliance", 
         "repairs", "education", "business", "retraining", "other"],
        help="Reason for taking the loan"
    )
    
    # Additional Information
    st.sidebar.markdown("### 📝 Additional Information")
    residence_since = st.sidebar.slider("Residence Since (years)", 1, 4, 2)
    num_dependents = st.sidebar.slider("Number of Dependents", 1, 2, 1)
    existing_credits = st.sidebar.slider("Existing Credits", 1, 4, 1)
    
    personal_status = st.sidebar.selectbox(
        "Personal Status & Gender",
        ["male div/sep", "female div/dep/mar", "male single", "male mar/wid", "female single"]
    )
    
    other_parties = st.sidebar.selectbox(
        "Other Parties",
        ["none", "co applicant", "guarantor"]
    )
    
    property_magnitude = st.sidebar.selectbox(
        "Property",
        ["real estate", "life insurance", "car", "no known property"]
    )
    
    other_payment_plans = st.sidebar.selectbox(
        "Other Payment Plans",
        ["bank", "stores", "none"]
    )
    
    housing = st.sidebar.selectbox(
        "Housing",
        ["rent", "own", "for free"]
    )
    
    own_telephone = st.sidebar.selectbox("Own Telephone", ["none", "yes"])
    foreign_worker = st.sidebar.selectbox("Foreign Worker", ["yes", "no"])
    
    # Create feature dictionary
    features = {
        'age': age,
        'credit_amount': credit_amount,
        'duration': duration,
        'installment_rate': installment_rate,
        'checking_status': checking_status,
        'savings_status': savings_status,
        'credit_history': credit_history,
        'employment': employment,
        'job': job,
        'purpose': purpose,
        'residence_since': residence_since,
        'num_dependents': num_dependents,
        'existing_credits': existing_credits,
        'personal_status': personal_status,
        'other_parties': other_parties,
        'property_magnitude': property_magnitude,
        'other_payment_plans': other_payment_plans,
        'housing': housing,
        'own_telephone': own_telephone,
        'foreign_worker': foreign_worker
    }
    
    return features

def preprocess_input(features, feature_names, scaler):
    """
    Preprocess input features to match training data format
    Applies one-hot encoding and feature scaling.
    """
    # Create DataFrame with one row
    df = pd.DataFrame([features])
    
    # Identify numerical and categorical columns before encoding
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Apply one-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Ensure all expected features are present
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Select and order columns to match training data
    df_encoded = df_encoded[feature_names]
    
    # Apply scaling to numerical features (same as training)
    numerical_feature_names = [col for col in df_encoded.columns if col in numerical_cols]
    if len(numerical_feature_names) > 0:
        df_encoded[numerical_feature_names] = scaler.transform(df_encoded[numerical_feature_names])
    
    return df_encoded

def display_prediction(prediction, probability):
    """Display prediction result with visual styling"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if prediction == 1:
            st.markdown("""
                <div class="success-box" style="text-align: center;">
                    <h1 style="color: #28a745; margin: 0;">✅ APPROVED</h1>
                    <h3 style="color: #155724; margin-top: 10px;">Good Credit Risk</h3>
                </div>
            """, unsafe_allow_html=True)
            st.success(f"**Confidence:** {probability:.1%}")
            st.balloons()
        else:
            st.markdown("""
                <div class="danger-box" style="text-align: center;">
                    <h1 style="color: #dc3545; margin: 0;">❌ REJECTED</h1>
                    <h3 style="color: #721c24; margin-top: 10px;">Bad Credit Risk</h3>
                </div>
            """, unsafe_allow_html=True)
            st.error(f"**Confidence:** {probability:.1%}")

def display_shap_explanation(model, explainer, input_data, feature_names):
    """Display SHAP-based explanations"""
    st.markdown("<div class='subheader'>🔍 Explainability Analysis</div>", unsafe_allow_html=True)
    
    # Calculate SHAP values
    with st.spinner("Generating SHAP explanations..."):
        shap_values = explainer.shap_values(input_data)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["📊 Feature Impact", "💧 Waterfall Plot", "🔢 Detailed Breakdown"])
    
    with tab1:
        st.markdown("#### How features influenced this prediction")
        
        # Create SHAP bar plot for single instance
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_values[0]
        })
        shap_df['Absolute SHAP'] = np.abs(shap_df['SHAP Value'])
        shap_df = shap_df.sort_values('Absolute SHAP', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if x < 0 else 'green' for x in shap_df['SHAP Value']]
        ax.barh(range(len(shap_df)), shap_df['SHAP Value'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(shap_df)))
        ax.set_yticklabels(shap_df['Feature'], fontsize=10)
        ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12, fontweight='bold')
        ax.set_title('Top 10 Features Influencing This Decision', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Interpretation:**
        - 🟢 Green bars: Push prediction towards **Good Credit**
        - 🔴 Red bars: Push prediction towards **Bad Credit**
        - Longer bars = Stronger influence
        """)
    
    with tab2:
        st.markdown("#### Step-by-step contribution to the final prediction")
        
        # SHAP waterfall plot
        shap_explanation = shap.Explanation(
            values=shap_values[0], 
            base_values=explainer.expected_value,
            data=input_data.values[0],
            feature_names=feature_names
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.waterfall_plot(shap_explanation, max_display=12, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.info("The waterfall plot shows how the prediction starts from the base value (average model output) and each feature pushes it up or down to reach the final prediction.")
    
    with tab3:
        st.markdown("#### Detailed feature contributions")
        
        # Create detailed breakdown table
        detail_df = pd.DataFrame({
            'Feature': feature_names,
            'Feature Value': input_data.values[0],
            'SHAP Value': shap_values[0],
            'Impact': ['Positive ↑' if x > 0 else 'Negative ↓' for x in shap_values[0]]
        })
        detail_df['Abs SHAP'] = np.abs(detail_df['SHAP Value'])
        detail_df = detail_df.sort_values('Abs SHAP', ascending=False).head(15)
        detail_df = detail_df[['Feature', 'Feature Value', 'SHAP Value', 'Impact']]
        detail_df['SHAP Value'] = detail_df['SHAP Value'].round(4)
        
        st.dataframe(
            detail_df,
            use_container_width=True,
            hide_index=True
        )

def display_lime_explanation(model, lime_explainer, input_data, feature_names, prediction):
    """Display LIME-based explanations"""
    st.markdown("<div class='subheader'>🔬 LIME Local Explanation</div>", unsafe_allow_html=True)
    
    st.markdown("""
    **LIME (Local Interpretable Model-agnostic Explanations)** creates a simple, interpretable model 
    around this specific prediction to explain the decision locally.
    """)
    
    # Generate LIME explanation
    with st.spinner("Generating LIME explanation..."):
        try:
            lime_exp = lime_explainer.explain_instance(
                input_data.values[0],
                model.predict_proba,
                num_features=10,
                top_labels=2
            )
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["📊 Top Features", "📈 Prediction Probabilities", "💡 Interpretation"])
            
            with tab1:
                st.markdown("#### Top 10 features influencing this prediction")
                
                # Get explanation for the predicted class
                exp_list = lime_exp.as_list(label=prediction)
                
                # Create DataFrame for visualization
                lime_df = pd.DataFrame(exp_list, columns=['Feature Condition', 'Weight'])
                lime_df['Absolute Weight'] = np.abs(lime_df['Weight'])
                
                # Create bar plot
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['green' if x > 0 else 'red' for x in lime_df['Weight']]
                ax.barh(range(len(lime_df)), lime_df['Weight'], color=colors, alpha=0.7)
                ax.set_yticks(range(len(lime_df)))
                ax.set_yticklabels(lime_df['Feature Condition'], fontsize=9)
                ax.set_xlabel('Feature Weight (Impact on Prediction)', fontsize=12, fontweight='bold')
                ax.set_title('LIME Feature Importance', fontsize=14, fontweight='bold')
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
                ax.grid(axis='x', alpha=0.3)
                plt.gca().invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("""
                **Interpretation:**
                - 🟢 Green bars: Support the predicted class
                - 🔴 Red bars: Oppose the predicted class
                - Feature conditions show the actual value ranges
                """)
            
            with tab2:
                st.markdown("#### Model's confidence in each class")
                
                # Get probabilities
                probs = lime_exp.predict_proba
                
                # Create probability visualization
                fig, ax = plt.subplots(figsize=(8, 4))
                classes = ['Bad Credit\n(Reject)', 'Good Credit\n(Approve)']
                colors_prob = ['#ff6b6b', '#51cf66']
                bars = ax.bar(classes, probs, color=colors_prob, alpha=0.7, edgecolor='black', linewidth=2)
                
                # Add percentage labels on bars
                for bar, prob in zip(bars, probs):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{prob*100:.1f}%',
                           ha='center', va='bottom', fontweight='bold', fontsize=12)
                
                ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
                ax.set_title('Prediction Probabilities by Class', fontsize=14, fontweight='bold')
                ax.set_ylim(0, 1.1)
                ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display probabilities as metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Bad Credit Probability", f"{probs[0]*100:.2f}%")
                with col2:
                    st.metric("Good Credit Probability", f"{probs[1]*100:.2f}%")
            
            with tab3:
                st.markdown("#### Detailed LIME Explanation")
                
                # Create detailed table
                lime_detail_df = pd.DataFrame(exp_list, columns=['Feature Condition', 'Weight'])
                lime_detail_df['Impact'] = ['Supports Prediction ↑' if x > 0 else 'Opposes Prediction ↓' 
                                            for x in lime_detail_df['Weight']]
                lime_detail_df['Weight'] = lime_detail_df['Weight'].round(4)
                
                st.dataframe(
                    lime_detail_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                st.info("""
                **How LIME Works:**
                1. Creates synthetic data points around this specific instance
                2. Trains a simple linear model on these points
                3. Uses the simple model to explain the complex model's decision
                4. Shows which feature ranges most influenced this particular prediction
                """)
                
                # Show intercept (base prediction)
                intercept = lime_exp.intercept[prediction]
                st.markdown(f"**Base prediction (intercept):** {intercept:.4f}")
                st.markdown(f"**Final prediction score:** {sum([w for _, w in exp_list]) + intercept:.4f}")
        
        except Exception as e:
            st.error(f"Error generating LIME explanation: {str(e)}")
            st.info("LIME explanations are computed on-demand and may occasionally fail for edge cases.")

def display_risk_factors(shap_values, feature_names, input_data, prediction):
    """Display key risk factors and recommendations"""
    st.markdown("<div class='subheader'>⚠️ Key Risk Factors & Recommendations</div>", unsafe_allow_html=True)
    
    # Get top contributing features
    contrib_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_values[0],
        'Value': input_data.values[0]
    })
    contrib_df['Abs_SHAP'] = np.abs(contrib_df['SHAP Value'])
    
    if prediction == 0:  # Bad credit
        # Get factors pushing towards rejection
        negative_factors = contrib_df[contrib_df['SHAP Value'] < 0].sort_values('SHAP Value').head(5)
        
        st.markdown("#### Main reasons for rejection:")
        for idx, row in negative_factors.iterrows():
            st.markdown(f"""
            <div class="warning-box">
                <strong>🔸 {row['Feature'].replace('_', ' ').title()}</strong><br>
                Impact: {row['SHAP Value']:.3f} (High Risk Factor)
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### 💡 Recommendations to improve credit score:")
        st.markdown("""
        - ✅ Build a positive credit history by making timely payments
        - ✅ Increase savings account balance
        - ✅ Consider a smaller loan amount or shorter duration
        - ✅ Maintain stable employment
        - ✅ Keep checking account balance positive
        """)
    else:  # Good credit
        # Get factors supporting approval
        positive_factors = contrib_df[contrib_df['SHAP Value'] > 0].sort_values('SHAP Value', ascending=False).head(5)
        
        st.markdown("#### Main reasons for approval:")
        for idx, row in positive_factors.iterrows():
            st.markdown(f"""
            <div class="success-box">
                <strong>🔸 {row['Feature'].replace('_', ' ').title()}</strong><br>
                Impact: {row['SHAP Value']:.3f} (Positive Factor)
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### 💡 Advice to maintain good credit:")
        st.markdown("""
        - ✅ Continue making timely payments
        - ✅ Maintain stable employment and income
        - ✅ Avoid taking on too much debt
        - ✅ Keep savings for emergencies
        - ✅ Monitor credit report regularly
        """)

def main():
    """Main application logic"""
    
    # Header
    st.markdown("<div class='main-header'>💳 Explainable Credit Risk Prediction System</div>", unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
        <div class="info-box">
        <strong>🎯 About This Application</strong><br>
        This system uses advanced Machine Learning (XGBoost) combined with Explainable AI (SHAP & LIME) 
        to assess credit risk transparently. Every decision is explained, ensuring regulatory compliance 
        and building trust with stakeholders.
        </div>
    """, unsafe_allow_html=True)
    
    # Load models
    model, scaler, feature_names, shap_explainer, X_train_sample, models_loaded = load_models()
    
    if not models_loaded:
        st.stop()
    
    # Create LIME explainer (on-demand, can't be pickled)
    lime_explainer = create_lime_explainer(X_train_sample, feature_names)
    
    # Get input features
    features = create_input_features()
    
    # Predict button
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🚀 Predict Credit Risk", type="primary", use_container_width=True)
    
    if predict_button:
        st.session_state.prediction_made = True
        
        # Preprocess input
        with st.spinner("Processing application..."):
            input_data = preprocess_input(features, feature_names, scaler)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        confidence = probability[prediction]
        
        # Store in session state
        st.session_state.prediction = prediction
        st.session_state.probability = confidence
        st.session_state.input_data = input_data
        st.session_state.features = features
    
    # Display results if prediction was made
    if st.session_state.prediction_made:
        # Display prediction
        display_prediction(st.session_state.prediction, st.session_state.probability)
        
        st.markdown("---")
        
        # Display SHAP explanations
        display_shap_explanation(
            model, 
            shap_explainer, 
            st.session_state.input_data, 
            feature_names
        )
        
        st.markdown("---")
        
        # Display LIME explanations
        display_lime_explanation(
            model,
            lime_explainer,
            st.session_state.input_data,
            feature_names,
            st.session_state.prediction
        )
        
        st.markdown("---")
        
        # Display risk factors and recommendations
        shap_values = shap_explainer.shap_values(st.session_state.input_data)
        display_risk_factors(
            shap_values, 
            feature_names, 
            st.session_state.input_data, 
            st.session_state.prediction
        )
        
        st.markdown("---")
        
        # Application summary
        with st.expander("📋 View Application Summary"):
            st.json(st.session_state.features)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
        <p><strong>Explainable Credit Risk Prediction System</strong></p>
        <p>Powered by XGBoost + SHAP + LIME | Compliant with GDPR, Basel III, and EU AI Act</p>
        <p>© 2026 | For demonstration purposes</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
