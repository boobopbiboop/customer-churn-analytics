import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar - Profile
with st.sidebar:
    st.image("https://via.placeholder.com/150x150/667eea/white?text=CA", width=150)
    st.markdown("### 👤 **Profil**")
    st.write("**Nama:** Nadya Saraswati Putri")
    st.write("**LinkedIn:** www.linkedin.com/in/nadyasaraswatip")
    st.write("**GitHub:** https://github.com/boobopbiboop")
    
    st.markdown("---")
    st.markdown("### 🎯 **Navigasi**")
    page = st.selectbox("Pilih Halaman:", 
                       ["📈 Overview", "🔍 EDA", "🤖 Machine Learning", "💡 Insights"])

# Load and prepare data function
@st.cache_data
def load_data():
    df = pd.read_csv('churn.csv')
    return df

# Main Header
st.markdown('<h1 class="main-header">📊 Customer Churn Analytics Dashboard</h1>', 
            unsafe_allow_html=True)

# Load data
df = load_data()

# Page routing
if page == "📈 Overview":
    st.markdown('<div class="sub-header">🎯 Project Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>🎯 Tujuan Proyek</h4>
        Menganalisis pola customer churn pada perusahaan telekomunikasi untuk:
        <ul>
        <li>Mengidentifikasi faktor-faktor penyebab churn</li>
        <li>Membangun model prediksi churn yang akurat</li>
        <li>Memberikan rekomendasi strategis untuk retensi pelanggan</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Dataset Summary")
        st.metric("Total Customers", f"{len(df):,}")
        st.metric("Features", f"{len(df.columns)-1}")
        st.metric("Churn Rate", f"{(df['Churn']=='Yes').mean():.1%}")
    
    # Key Metrics
    st.markdown('<div class="sub-header">📈 Key Performance Indicators</div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(df)
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        churned_customers = (df['Churn'] == 'Yes').sum()
        st.metric("Churned Customers", f"{churned_customers:,}")
    
    with col3:
        avg_tenure = df['Tenure'].mean()
        st.metric("Avg Tenure", f"{avg_tenure:.1f} months")
    
    with col4:
        avg_charges = df['MonthlyCharges'].mean()
        st.metric("Avg Monthly Charges", f"${avg_charges:.2f}")

elif page == "🔍 EDA":
    st.markdown('<div class="sub-header">🔍 Exploratory Data Analysis</div>', 
                unsafe_allow_html=True)
    
    # Churn Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(df, names='Churn', title='Customer Churn Distribution',
                        color_discrete_map={'Yes': '#e74c3c', 'No': '#3498db'})
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        churn_gender = pd.crosstab(df['Gender'], df['Churn'], normalize='index') * 100
        fig_bar = px.bar(churn_gender, 
                        title='Churn Rate by Gender (%)',
                        color_discrete_map={'Yes': '#e74c3c', 'No': '#3498db'})
        fig_bar.update_layout(xaxis_title='Gender', yaxis_title='Percentage (%)')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Financial Analysis
    st.markdown("### 💰 Financial Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_box1 = px.box(df, x='Churn', y='MonthlyCharges', 
                         title='Monthly Charges Distribution by Churn',
                         color='Churn',
                         color_discrete_map={'Yes': '#e74c3c', 'No': '#3498db'})
        st.plotly_chart(fig_box1, use_container_width=True)
    
    with col2:
        fig_box2 = px.box(df, x='Churn', y='Tenure', 
                         title='Tenure Distribution by Churn',
                         color='Churn',
                         color_discrete_map={'Yes': '#e74c3c', 'No': '#3498db'})
        st.plotly_chart(fig_box2, use_container_width=True)
    
    # Contract Analysis
    st.markdown("### 📋 Contract Analysis")
    contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
    fig_contract = px.bar(contract_churn, 
                         title='Churn Rate by Contract Type (%)',
                         color_discrete_map={'Yes': '#e74c3c', 'No': '#3498db'})
    fig_contract.update_layout(xaxis_title='Contract Type', yaxis_title='Percentage (%)')
    st.plotly_chart(fig_contract, use_container_width=True)
    
    # Interactive Filter
    st.markdown("### 🎛️ Interactive Analysis")
    selected_feature = st.selectbox("Choose feature to analyze:", 
                                   ['Gender', 'Partner', 'Dependents', 'Contract', 
                                    'PaperlessBilling', 'PaymentMethod'])
    
    feature_churn = pd.crosstab(df[selected_feature], df['Churn'], normalize='index') * 100
    fig_interactive = px.bar(feature_churn, 
                           title=f'Churn Rate by {selected_feature} (%)',
                           color_discrete_map={'Yes': '#e74c3c', 'No': '#3498db'})
    st.plotly_chart(fig_interactive, use_container_width=True)

# elif page == "🤖 Machine Learning":
#     st.markdown('<div class="sub-header">🤖 Machine Learning Analysis</div>', 
#                 unsafe_allow_html=True)
    
#     # Data Preparation
#     X = df.drop(['Churn', 'customerID'], axis=1)
#     y = df['Churn']
    
#     # Encoding
#     le_target = LabelEncoder()
#     y_encoded = le_target.fit_transform(y)
    
#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
#     )
    
#     # Feature engineering
#     numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
#     categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    
#     # Simple preprocessing for demo
#     from sklearn.preprocessing import LabelEncoder as LE
#     X_train_processed = X_train.copy()
#     X_test_processed = X_test.copy()
    
#     for col in categorical_features:
#         le = LE()
#         X_train_processed[col] = le.fit_transform(X_train_processed[col])
#         X_test_processed[col] = le.transform(X_test_processed[col])
    
#     # Model Training
#     st.markdown("### 🎯 Model Performance")
    
#     models = {
#         'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
#         'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
#         'Gradient Boosting': GradientBoostingClassifier(random_state=42)
#     }
    
#     results = {}
    
#     for name, model in models.items():
#         model.fit(X_train_processed, y_train)
#         y_pred = model.predict(X_test_processed)
#         y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
        
#         roc_auc = roc_auc_score(y_test, y_pred_proba)
#         results[name] = {
#             'ROC-AUC': roc_auc,
#             'model': model,
#             'predictions': y_pred,
#             'probabilities': y_pred_proba
#         }
    
#     # Display results
#     results_df = pd.DataFrame({
#         'Model': list(results.keys()),
#         'ROC-AUC': [results[model]['ROC-AUC'] for model in results.keys()]
#     }).sort_values('ROC-AUC', ascending=False)
    
#     col1, col2 = st.columns([1, 2])
    
#     with col1:
#         st.markdown("#### 🏆 Model Ranking")
#         results_df['Rank'] = range(1, len(results_df) + 1)
#         st.dataframe(results_df.round(4))
    
#     with col2:
#         # ROC Curves
#         fig_roc = go.Figure()
        
#         for name in results.keys():
#             fpr, tpr, _ = roc_curve(y_test, results[name]['probabilities'])
#             fig_roc.add_trace(go.Scatter(
#                 x=fpr, y=tpr,
#                 name=f'{name} (AUC = {results[name]["ROC-AUC"]:.3f})',
#                 line=dict(width=2)
#             ))
        
#         fig_roc.add_trace(go.Scatter(
#             x=[0, 1], y=[0, 1],
#             mode='lines',
#             name='Random Classifier',
#             line=dict(dash='dash', color='gray')
#         ))
        
#         fig_roc.update_layout(
#             title='ROC Curves Comparison',
#             xaxis_title='False Positive Rate',
#             yaxis_title='True Positive Rate',
#             width=600, height=400
#         )
#         st.plotly_chart(fig_roc, use_container_width=True)
    
#     # Best Model Analysis
#     best_model_name = results_df.iloc[0]['Model']
#     best_model = results[best_model_name]['model']
    
#     st.markdown(f"### 🌟 Best Model: {best_model_name}")
    
#     # Feature Importance
#     if hasattr(best_model, 'feature_importances_'):
#         importances = best_model.feature_importances_
#         feature_names = X_train_processed.columns
        
#         feature_importance_df = pd.DataFrame({
#             'Feature': feature_names,
#             'Importance': importances
#         }).sort_values('Importance', ascending=False).head(10)
        
#         fig_importance = px.bar(
#             feature_importance_df, 
#             x='Importance', y='Feature',
#             orientation='h',
#             title='Top 10 Feature Importance',
#             color='Importance',
#             color_continuous_scale='viridis'
#         )
#         fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
#         st.plotly_chart(fig_importance, use_container_width=True)
    
#     # Confusion Matrix
#     from sklearn.metrics import accuracy_score
#     y_pred_best = results[best_model_name]['predictions']
#     cm = confusion_matrix(y_test, y_pred_best)
    
#     fig_cm = px.imshow(cm, 
#                       text_auto=True, 
#                       aspect="auto",
#                       title='Confusion Matrix',
#                       labels=dict(x="Predicted", y="Actual"),
#                       x=['No Churn', 'Churn'],
#                       y=['No Churn', 'Churn'])
#     st.plotly_chart(fig_cm, use_container_width=True)
    
#     # Model Metrics
#     accuracy = accuracy_score(y_test, y_pred_best)
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.metric("Accuracy", f"{accuracy:.3f}")
#     with col2:
#         st.metric("ROC-AUC", f"{results[best_model_name]['ROC-AUC']:.3f}")
#     with col3:
#         st.metric("Test Samples", len(y_test))

elif page == "🤖 Machine Learning":
    st.markdown('<div class="sub-header">🤖 Machine Learning Analysis</div>', 
                unsafe_allow_html=True)
    
    # Data Quality Check
    st.markdown("### 🔍 Data Quality Assessment")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        missing_count = df.isnull().sum().sum()
        st.metric("Missing Values", missing_count)
    
    with col2:
        duplicate_count = df.duplicated().sum()
        st.metric("Duplicate Rows", duplicate_count)
    
    with col3:
        total_features = len(df.columns) - 2  # excluding Churn and customerID
        st.metric("Features for Modeling", total_features)
    
    # Data Preparation
    X = df.drop(['Churn', 'customerID'], axis=1).copy()
    y = df['Churn'].copy()
    
    # Handle missing values BEFORE train-test split
    st.markdown("### 🔧 Data Preprocessing")
    
    # Check for missing values and handle them
    missing_info = X.isnull().sum()
    if missing_info.sum() > 0:
        st.warning(f"Found {missing_info.sum()} missing values. Applying preprocessing...")
        
        # Handle missing values
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        # Fill numerical missing values with median
        for col in numerical_cols:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                X[col].fillna(median_val, inplace=True)
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                mode_val = X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown'
                X[col].fillna(mode_val, inplace=True)
    
    # Encoding target variable
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    # Display preprocessing results
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"✅ Missing values handled: {X.isnull().sum().sum()} remaining")
    with col2:
        st.info(f"✅ Target encoded: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Feature identification
    numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    
    st.markdown(f"**Numerical Features ({len(numerical_features)}):** {', '.join(numerical_features)}")
    st.markdown(f"**Categorical Features ({len(categorical_features)}):** {', '.join(categorical_features)}")
    
    # Advanced preprocessing with proper encoding
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    # Store encoders for consistent transformation
    label_encoders = {}
    
    # Encode categorical variables
    for col in categorical_features:
        le = LabelEncoder()
        X_train_processed[col] = le.fit_transform(X_train_processed[col].astype(str))
        X_test_processed[col] = le.transform(X_test_processed[col].astype(str))
        label_encoders[col] = le
    
    # Verify no missing values remain
    train_missing = X_train_processed.isnull().sum().sum()
    test_missing = X_test_processed.isnull().sum().sum()
    
    if train_missing > 0 or test_missing > 0:
        st.error(f"❌ Still have missing values - Train: {train_missing}, Test: {test_missing}")
        st.stop()
    
    # Model Training
    st.markdown("### 🎯 Model Performance")
    
    # Use a progress bar for model training
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    
    for i, (name, model) in enumerate(models.items(), 1):
        try:
            # Progress update
            progress = i / len(models)
            progress_bar.progress(progress)
            status_text.text(f"Training {name}... ({i}/{len(models)})")
            
            # Train model
            model.fit(X_train_processed, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_processed)
            y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
            
            # Calculate metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Store results
            results[name] = {
                'ROC-AUC': roc_auc,
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
        except Exception as e:
            st.error(f"Error training {name}: {str(e)}")
            continue
    
    # Pastikan results tidak kosong
    if not results:
        st.error("❌ Tidak ada model yang berhasil di-training!")
        st.stop()
    
    # Display results
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'ROC-AUC': [results[model]['ROC-AUC'] for model in results.keys()]
    }).sort_values('ROC-AUC', ascending=False)

    # for name, model in models.items():
    #     model.fit(X_train_processed, y_train)
    #     y_pred = model.predict(X_test_processed)
    #     y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
        
    #     roc_auc = roc_auc_score(y_test, y_pred_proba)
        
    # # Display results
    # results_df = pd.DataFrame({
    #     'Model': list(results.keys()),
    #     'ROC-AUC': [results[model]['ROC-AUC'] for model in results.keys()]
    # }).sort_values('ROC-AUC', ascending=False)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 🏆 Model Ranking")
        results_df['Rank'] = range(1, len(results_df) + 1)
        st.dataframe(results_df.round(4))
    
    with col2:
        # ROC Curves
        fig_roc = go.Figure()
        
        for name in results.keys():
            fpr, tpr, _ = roc_curve(y_test, results[name]['probabilities'])
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f'{name} (AUC = {results[name]["ROC-AUC"]:.3f})',
                line=dict(width=2)
            ))
        
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        fig_roc.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600, height=400
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    
    # Best Model Analysis
    best_model_name = results_df.iloc[0]['Model']
    best_model = results[best_model_name]['model']
    
    st.markdown(f"### 🌟 Best Model: {best_model_name}")
    
    # Feature Importance
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_names = X_train_processed.columns
        
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(10)
        
        fig_importance = px.bar(
            feature_importance_df, 
            x='Importance', y='Feature',
            orientation='h',
            title='Top 10 Feature Importance',
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Confusion Matrix
    from sklearn.metrics import accuracy_score
    y_pred_best = results[best_model_name]['predictions']
    cm = confusion_matrix(y_test, y_pred_best)
    
    fig_cm = px.imshow(cm, 
                      text_auto=True, 
                      aspect="auto",
                      title='Confusion Matrix',
                      labels=dict(x="Predicted", y="Actual"),
                      x=['No Churn', 'Churn'],
                      y=['No Churn', 'Churn'])
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Model Metrics
    accuracy = accuracy_score(y_test, y_pred_best)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    with col2:
        st.metric("ROC-AUC", f"{results[best_model_name]['ROC-AUC']:.3f}")
    with col3:
        st.metric("Test Samples", len(y_test))

elif page == "💡 Insights":
    st.markdown('<div class="sub-header">💡 Business Insights & Recommendations</div>', 
                unsafe_allow_html=True)
    
    # Key Findings
    st.markdown("""
    <div class="insight-box">
    <h4>🔍 Key Findings</h4>
    
    <h5>1. Customer Demographics:</h5>
    <ul>
    <li>Churn rate sangat tinggi pada pelanggan dengan kontrak month-to-month (~50%)</li>
    <li>Pelanggan dengan tenure rendah (<12 bulan) memiliki kecenderungan churn yang tinggi</li>
    <li>Senior citizens menunjukkan churn rate yang lebih tinggi dibanding non-senior</li>
    </ul>
    
    <h5>2. Financial Patterns:</h5>
    <ul>
    <li>Pelanggan dengan monthly charges tinggi (>$70) memiliki churn rate lebih tinggi</li>
    <li>Pelanggan tanpa partner/dependents lebih cenderung churn</li>
    <li>Electronic check payment method berkorelasi dengan tingkat churn yang tinggi</li>
    </ul>
    
    <h5>3. Service Characteristics:</h5>
    <ul>
    <li>Paperless billing customers menunjukkan churn rate yang lebih tinggi</li>
    <li>Model machine learning mencapai ROC-AUC score 0.85+ dalam prediksi churn</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Strategic Recommendations
    st.markdown("### 🎯 Strategic Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>🛡️ Retention Strategies</h4>
        <ol>
        <li><strong>Contract Incentives:</strong> Berikan diskon untuk kontrak annual/2-year</li>
        <li><strong>Early Warning System:</strong> Monitor pelanggan dengan tenure <6 bulan</li>
        <li><strong>Payment Method:</strong> Insentif untuk beralih dari electronic check</li>
        <li><strong>Senior Citizen Program:</strong> Program khusus untuk pelanggan senior</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4>📊 Operational Improvements</h4>
        <ol>
        <li><strong>Pricing Strategy:</strong> Review pricing untuk high-value customers</li>
        <li><strong>Customer Onboarding:</strong> Improve experience untuk new customers</li>
        <li><strong>Communication:</strong> Proactive outreach untuk at-risk segments</li>
        <li><strong>Service Quality:</strong> Focus pada paperless billing experience</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # ROI Calculation
    st.markdown("### 💰 Potential Business Impact")
    
    # Simulasi perhitungan ROI
    total_customers = len(df)
    churn_rate = (df['Churn'] == 'Yes').mean()
    avg_monthly_revenue = df['MonthlyCharges'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        monthly_churn_cost = total_customers * churn_rate * avg_monthly_revenue
        st.metric("Monthly Churn Cost", f"${monthly_churn_cost:,.0f}")
    
    with col2:
        annual_churn_cost = monthly_churn_cost * 12
        st.metric("Annual Churn Cost", f"${annual_churn_cost:,.0f}")
    
    with col3:
        potential_reduction = 0.2  # 20% reduction in churn
        savings = annual_churn_cost * potential_reduction
        st.metric("Potential Annual Savings", f"${savings:,.0f}")
    
    with col4:
        roi_multiple = savings / 100000  # Assuming $100k investment
        st.metric("ROI Multiple", f"{roi_multiple:.1f}x")
    
    # Implementation Roadmap
    st.markdown("### 🗺️ Implementation Roadmap")
    
    roadmap_data = {
        'Phase': ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4'],
        'Timeline': ['Month 1-2', 'Month 3-4', 'Month 5-6', 'Month 7-12'],
        'Focus': [
            'Data Infrastructure & Model Deployment',
            'High-Risk Customer Identification',
            'Retention Campaign Implementation', 
            'Optimization & Scaling'
        ],
        'Expected Impact': ['10%', '15%', '20%', '25%']
    }
    
    roadmap_df = pd.DataFrame(roadmap_data)
    st.dataframe(roadmap_df, use_container_width=True)
    
    # Success Metrics
    st.markdown("""
    <div class="insight-box">
    <h4>📈 Success Metrics to Track</h4>
    <ul>
    <li><strong>Primary:</strong> Monthly churn rate reduction (target: 20% decrease)</li>
    <li><strong>Secondary:</strong> Customer lifetime value increase</li>
    <li><strong>Operational:</strong> Early warning system accuracy (>85%)</li>
    <li><strong>Financial:</strong> Cost per retained customer vs acquisition cost</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>📊 Customer Churn Analytics Dashboard | Built with Streamlit</p>
    <p>Data Science Project - Nadya SP | 2025</p>
            <p> https://github.com/boobopbiboop/customer-churn-analytics</p>
</div>
</div>
""", unsafe_allow_html=True)