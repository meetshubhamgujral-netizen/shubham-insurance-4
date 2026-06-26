import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# 1. Page Configuration & Theme Settings
st.set_page_config(
    page_title="Claims Bias & Predictive Analytics Dashboard",
    layout="wide",
    page_icon="⚖️"
)

# Custom colorful professional styling
st.markdown("""
    <style>
    .main-header {
        background-color: #1E3A8A;
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
    }
    .metric-box {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 15px;
    }
    .bias-alert {
        background-color: #FEF2F2;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #EF4444;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Data Loading & Cleaning Functions
@st.cache_data
def load_and_clean_data(filepath="Insurance.csv"):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return None

    for col in ['SUM_ASSURED', 'PI_ANNUAL_INCOME']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    if 'PI_OCCUPATION' in df.columns:
        df['PI_OCCUPATION'] = df['PI_OCCUPATION'].fillna('Unknown')
    if 'REASON_FOR_CLAIM' in df.columns:
        df['REASON_FOR_CLAIM'] = df['REASON_FOR_CLAIM'].fillna('Not Disclosed')
        
    return df

# Main Header
st.markdown('<div class="main-header"><h1>⚖️ Insurance Claim Settlement Bias & Predictive Dashboard</h1><p>An Advanced Analytical Tool for Claim Officers to Detect Operational Biases and Train Super-Learning Classifiers</p></div>', unsafe_allow_html=True)

# File Management in Sidebar
st.sidebar.header("📁 Data Source Settings")
uploaded_file = st.sidebar.file_uploader("Upload your Insurance Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    for col in ['SUM_ASSURED', 'PI_ANNUAL_INCOME']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    if 'PI_OCCUPATION' in df.columns:
        df['PI_OCCUPATION'] = df['PI_OCCUPATION'].fillna('Unknown')
    if 'REASON_FOR_CLAIM' in df.columns:
        df['REASON_FOR_CLAIM'] = df['REASON_FOR_CLAIM'].fillna('Not Disclosed')
else:
    df = load_and_clean_data()

if df is None:
    st.error("⚠️ Insurance.csv file not found! Please place 'Insurance.csv' in the same folder or upload it via the sidebar.")
    st.stop()

# Basic KPIs
total_claims = len(df)
approved_count = len(df[df['POLICY_STATUS'] == 'Approved Death Claim'])
repudiated_count = len(df[df['POLICY_STATUS'] == 'Repudiate Death'])
approval_rate = (approved_count / total_claims) * 100 if total_claims > 0 else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric(label="📊 Total Claims Filed", value=f"{total_claims:,}")
with kpi2:
    st.metric(label="✅ Approved Claims", value=f"{approved_count:,}", delta=f"{approval_rate:.1f}% Rate")
with kpi3:
    st.metric(label="❌ Repudiated Claims", value=f"{repudiated_count:,}", delta=f"{100 - approval_rate:.1f}% Rate", delta_color="inverse")
with kpi4:
    st.metric(label="💰 Average Sum Assured", value=f"₹{df['SUM_ASSURED'].mean():,.2f}")

# Initialize session state for tracking model results dynamically
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = None

# 3. Create Operational Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Descriptive Cross-Tabs", 
    "🔍 Diagnostic Bias Probing", 
    "🤖 Super-Learning ML Models", 
    "💡 Executive Insights & Findings"
])

# ================= TAB 1: DESCRIPTIVE ANALYTICS =================
with tab1:
    st.header("📋 Cross-Tabulation Analytics against Policy Status")
    st.write("Cross-tabulating categorical attributes directly evaluates how policy frequencies shift across different application statuses.")
    
    categorical_options = [col for col in ['PI_GENDER', 'ZONE', 'PAYMENT_MODE', 'EARLY_NON', 'MEDICAL_NONMED', 'PI_STATE'] if col in df.columns]
    selected_cat = st.selectbox("Select Feature for Cross-Tabulation Analysis:", categorical_options)
    
    if selected_cat:
        ct_count = pd.crosstab(df[selected_cat], df['POLICY_STATUS'])
        ct_pct = pd.crosstab(df[selected_cat], df['POLICY_STATUS'], normalize='index') * 100
        
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            st.subheader(f"Raw Headcount Cross-Tab: {selected_cat}")
            st.dataframe(ct_count.style.background_gradient(cmap='Blues', axis=0))
            
        with col_t2:
            st.subheader(f"Percentage Allocation Cross-Tab (%)")
            st.dataframe(ct_pct.style.format("{:.2f}%").background_gradient(cmap='YlOrRd', axis=1))
            
        st.subheader(f"Visualizing Claims Split across {selected_cat}")
        fig_bar = px.bar(
            df, 
            x=selected_cat, 
            color='POLICY_STATUS',
            barmode='group',
            color_discrete_sequence=['#10B981', '#EF4444'],
            title=f"Distribution of Policy Status by {selected_cat}"
        )
        fig_bar.update_layout(xaxis={'categoryorder':'total descending'}, height=500)
        st.plotly_chart(fig_bar, use_container_width=True)

# ================= TAB 2: DIAGNOSTIC ANALYTICS =================
with tab2:
    st.header("🔍 Diagnostic Bias Probe: Deep Analysis")
    
    diag_selection = st.radio("Choose Target Focus for Bias Probe:", ["Age-Wise Bias", "Income-Wise Bias", "Team/Zone-Wise Bias"], horizontal=True)
    
    if diag_selection == "Age-Wise Bias":
        st.subheader("👨‍🦳 Age-Wise Bias Inspection")
        df['AGE_GROUP'] = pd.cut(df['PI_AGE'], bins=[0, 30, 45, 60, 75, 100], labels=['<30', '30-45', '45-60', '60-75', '>75'])
        
        col_a1, col_a2 = st.columns([1, 1])
        with col_a1:
            fig_age_box = px.box(df, x='POLICY_STATUS', y='PI_AGE', color='POLICY_STATUS',
                                 color_discrete_sequence=['#10B981', '#EF4444'], points="all", title="Age Distribution vs Policy Status")
            st.plotly_chart(fig_age_box, use_container_width=True)
        with col_a2:
            age_ct = pd.crosstab(df['AGE_GROUP'], df['POLICY_STATUS'], normalize='index') * 100
            fig_age_bar = px.bar(age_ct.reset_index(), x='AGE_GROUP', y=age_ct.columns.tolist(),
                                 title="Approval/Repudiation Rate by Age Brackets (%)", color_discrete_sequence=['#10B981', '#EF4444'])
            st.plotly_chart(fig_age_bar, use_container_width=True)
            
        under_30_app = age_ct.loc['<30', 'Approved Death Claim'] if '<30' in age_ct.index and 'Approved Death Claim' in age_ct.columns else 0
        over_60_app = age_ct.loc['60-75', 'Approved Death Claim'] if '60-75' in age_ct.index and 'Approved Death Claim' in age_ct.columns else 0
        st.markdown(f"""
            <div class="bias-alert">
            <strong>Dynamic Age Insight:</strong> Policyholders under the age of 30 currently experience an approval rate of <b>{under_30_app:.1f}%</b>, 
            compared to older cohorts (60-75) sitting at an approval rate of <b>{over_60_app:.1f}%</b>.
            </div>
        """, unsafe_allow_html=True)
        
    elif diag_selection == "Income-Wise Bias":
        st.subheader("💰 Income-Wise Bias Inspection")
        
        mean_inc_approved = df[df['POLICY_STATUS'] == 'Approved Death Claim']['PI_ANNUAL_INCOME'].mean()
        mean_inc_repudiated = df[df['POLICY_STATUS'] == 'Repudiate Death']['PI_ANNUAL_INCOME'].mean()
        
        col_i1, col_i2 = st.columns([1, 1])
        with col_i1:
            avg_income = df.groupby('POLICY_STATUS')['PI_ANNUAL_INCOME'].mean().reset_index()
            fig_inc_bar = px.bar(avg_income, x='POLICY_STATUS', y='PI_ANNUAL_INCOME', color='POLICY_STATUS',
                                 color_discrete_sequence=['#3B82F6', '#F59E0B'], title="Mean Annual Income by Claim Outcome")
            st.plotly_chart(fig_inc_bar, use_container_width=True)
        with col_i2:
            df['HAS_REPORTED_INCOME'] = np.where(df['PI_ANNUAL_INCOME'] == 0, '0 / Unreported', 'Positive Income')
            inc_ct = pd.crosstab(df['HAS_REPORTED_INCOME'], df['POLICY_STATUS'], normalize='index') * 100
            fig_reported = px.bar(inc_ct.reset_index(), x='HAS_REPORTED_INCOME', y=inc_ct.columns.tolist(),
                                 title="Impact of Unreported/Zero Income Records (%)", color_discrete_sequence=['#10B981', '#EF4444'], barmode='group')
            st.plotly_chart(fig_reported, use_container_width=True)
            
        st.markdown(f"""
            <div class="bias-alert">
            <strong>Dynamic Income Insight:</strong> The dataset reveals that approved claims have an average income profile of <b>₹{mean_inc_approved:,.2f}</b>, 
            while rejected/repudiated claims show a significantly lower profile of <b>₹{mean_inc_repudiated:,.2f}</b>.
            </div>
        """, unsafe_allow_html=True)
        
    elif diag_selection == "Team/Zone-Wise Bias":
        st.subheader("🚩 Office Team & Zone Performance Disparities")
        zone_summary = pd.crosstab(df['ZONE'], df['POLICY_STATUS'], normalize='index') * 100
        if 'Repudiate Death' in zone_summary.columns:
            zone_summary = zone_summary.sort_values(by='Repudiate Death', ascending=False).reset_index()
        else:
            zone_summary = zone_summary.reset_index()
        
        fig_zone = px.bar(zone_summary, x='ZONE', y=zone_summary.columns.tolist()[1:], 
                          title="Claim Settlement Behaviors Across Teams & Zones (%)", color_discrete_sequence=['#10B981', '#EF4444'], height=600)
        st.plotly_chart(fig_zone, use_container_width=True)
        
        highest_rep_zone = zone_summary.iloc[0]['ZONE']
        highest_rep_val = zone_summary.iloc[0]['Repudiate Death'] if 'Repudiate Death' in zone_summary.columns else 0
        st.markdown(f"""
            <div class="bias-alert">
            <strong>Dynamic Zone Insight:</strong> High operational variance detected between regions. The team operating in <b>{highest_rep_zone}</b> 
            presents the most aggressive stance with a <b>{highest_rep_val:.1f}% claim repudiation rate</b>.
            </div>
        """, unsafe_allow_html=True)

# ================= TAB 3: MACHINE LEARNING & FEATURE ENGINEERING =================
with tab3:
    st.header("🤖 Machine Learning Super-Classifiers & Feature Engineering")
    
    if st.button("🚀 Execute Data Training & Evaluate Algorithms", type="primary"):
        with st.spinner("Engineering features and training super-learning algorithms..."):
            
            df_ml = df.copy()
            df_ml['INCOME_TO_SUM_ASSURED'] = df_ml['PI_ANNUAL_INCOME'] / (df_ml['SUM_ASSURED'] + 1)
            df_ml['AGE_GROUP'] = pd.cut(df_ml['PI_AGE'], bins=[0, 30, 45, 60, 75, 100], labels=['<30', '30-45', '45-60', '60-75', '>75']).astype(str)
            
            drop_cols = [c for c in ['POLICY_NO', 'PI_NAME'] if c in df_ml.columns]
            df_ml = df_ml.drop(columns=drop_cols)
            
            le_target = LabelEncoder()
            df_ml['POLICY_STATUS'] = le_target.fit_transform(df_ml['POLICY_STATUS'].astype(str))
            
            X = df_ml.drop(columns=['POLICY_STATUS'])
            y = df_ml['POLICY_STATUS']
            
            cat_cols = X.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            models = {
                'KNN': KNeighborsClassifier(n_neighbors=7),
                'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
                'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=8),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1)
            }
            
            metrics_list = []
            roc_curves = {}
            confusion_matrices = {}
            
            for name, model in models.items():
                X_tr = X_train_scaled if name == 'KNN' else X_train
                X_te = X_test_scaled if name == 'KNN' else X_test
                
                model.fit(X_tr, y_train)
                y_pred_tr = model.predict(X_tr)
                y_pred_te = model.predict(X_te)
                
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_te)[:, 1]
                else:
                    y_prob = model.decision_function(X_te)
                
                tr_acc = accuracy_score(y_train, y_pred_tr)
                te_acc = accuracy_score(y_test, y_pred_te)
                precision = precision_score(y_test, y_pred_te, zero_division=0)
                recall = recall_score(y_test, y_pred_te, zero_division=0)
                f1 = f1_score(y_test, y_pred_te, zero_division=0)
                
                metrics_list.append({
                    'Model': name, 'Train Accuracy': tr_acc, 'Test Accuracy': te_acc,
                    'Precision': precision, 'Recall': recall, 'F1 Score': f1
                })
                
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_curves[name] = (fpr, tpr, auc(fpr, tpr))
                confusion_matrices[name] = confusion_matrix(y_test, y_pred_te)
                
            res_df = pd.DataFrame(metrics_list)
            
            st.session_state.ml_results = {
                'df': res_df, 'roc': roc_curves, 'cm': confusion_matrices, 'classes': le_target.classes_
            }
            
        st.subheader("📈 Algorithm Performance Comparison Matrix")
        st.dataframe(res_df.style.format({
            'Train Accuracy': "{:.2%}", 'Test Accuracy': "{:.2%}",
            'Precision': "{:.2%}", 'Recall': "{:.2%}", 'F1 Score': "{:.2%}"
        }).background_gradient(cmap='Blues'))
        
        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            st.markdown("#### 🔄 Receiver Operating Characteristic (ROC) Curves")
            fig_roc = go.Figure()
            for name, (fpr, tpr, roc_auc) in roc_curves.items():
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC = {roc_auc:.2f})'))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='grey'), name='Random Guess'))
            fig_roc.update_layout(title="ROC Curve Model Stability Comparison", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=500)
            st.plotly_chart(fig_roc, use_container_width=True)
            
        with col_m2:
            st.markdown("#### 🧩 Confusion Matrices Grid")
            fig_cm, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes = axes.ravel()
            for idx, (name, cm) in enumerate(confusion_matrices.items()):
                sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Purples', xticklabels=le_target.classes_, yticklabels=le_target.classes_)
                axes[idx].set_title(f'{name} Matrix')
            plt.tight_layout()
            st.pyplot(fig_cm)

# ================= TAB 4: EXECUTIVE FINDINGS =================
with tab4:
    st.header("💡 Summary Executive Findings & Strategic Recommendations")
    
    df['AGE_GROUP'] = pd.cut(df['PI_AGE'], bins=[0, 30, 45, 60, 75, 100], labels=['<30', '30-45', '45-60', '60-75', '>75'])
    age_ct_live = pd.crosstab(df['AGE_GROUP'], df['POLICY_STATUS'], normalize='index') * 100
    zone_ct_live = pd.crosstab(df['ZONE'], df['POLICY_STATUS'], normalize='index') * 100
    med_ct_live = pd.crosstab(df['MEDICAL_NONMED'], df['POLICY_STATUS'], normalize='index') * 100
    
    mean_inc_app = df[df['POLICY_STATUS'] == 'Approved Death Claim']['PI_ANNUAL_INCOME'].mean()
    mean_inc_rep = df[df['POLICY_STATUS'] == 'Repudiate Death']['PI_ANNUAL_INCOME'].mean()
    
    st.subheader("📌 Identified Operational Disparities (Calculated from Current File)")
    
    if 'Repudiate Death' in zone_ct_live.columns and len(zone_ct_live) > 0:
        worst_zone = zone_ct_live.sort_values(by='Repudiate Death', ascending=False).index[0]
        worst_zone_val = zone_ct_live.sort_values(by='Repudiate Death', ascending=False)['Repudiate Death'].iloc[0]
        st.markdown(f"**1. Regional Variance Risk:** Extreme variance identified in claim handling across groups. The zone operating under **{worst_zone}** exhibits the highest rejection footprint, showing a **{worst_zone_val:.2f}% repudiation rate**.")
    else:
        st.markdown("**1. Regional Variance Risk:** Insufficient repudiation variance in current data entries to extract dynamic regional alerts.")
        
    st.markdown(f"**2. Socio-Economic Income Filter:** Approved claims maintain a mean annual income profile of **Hex-Value Formatted: ₹{mean_inc_app:,.2f}**, whereas rejected/repudiated records carry an average profile of **₹{mean_inc_rep:,.2f}**. Lower income accounts face a significantly higher statistical rate of rejection.")
    
    if 'MEDICAL' in med_ct_live.index and 'Approved Death Claim' in med_ct_live.columns:
        med_app_rate = med_ct_live.loc['MEDICAL', 'Approved Death Claim']
        non_med_app_rate = med_ct_live.loc['NON MEDICAL', 'Approved Death Claim'] if 'NON MEDICAL' in med_ct_live.index else 0
        st.markdown(f"**3. Underwriting Validation Disparity:** Policies processed with Medical Underwriting have an approval rate of **{med_app_rate:.2f}%**, while Non-Medical configurations drop to an approval rating of **{non_med_app_rate:.2f}%**.")
        
    st.subheader("🤖 Algorithmic Audit Summary")
    if st.session_state.ml_results is not None:
        res_df_saved = st.session_state.ml_results['df']
        best_model_row = res_df_saved.sort_values(by='Test Accuracy', ascending=False).iloc[0]
        st.info(f"**Active Model Insight:** According to the latest training run, the **{best_model_row['Model']}** model delivers the highest validation precision and capability, reaching a Test Accuracy of **{best_model_row['Test Accuracy']:.2%}** and an F1-Score of **{best_model_row['F1 Score']:.2%}**.")
    else:
        st.warning("⚠️ Run the Machine Learning pipelines under the 'Super-Learning ML Models' tab to see dynamic algorithmic insights here.")

    st.subheader("🛠完整 Operational Action Plan Recommendations")
    st.markdown("""
    * **Enforce Centralized Rulesets**: Enforce structured screening policies across regional hubs to normalize localized evaluation variances.
    * **Review Downstream Screening Filters**: Re-evaluate documentation procedures on non-medical applications to stop verification friction from flowing downstream to the claim settlement stage.
    * **Deploy Algorithmic Gatekeepers**: Use the best-performing super-learning algorithm to flag claim rejections that show high probabilities of structural bias for secondary manual audits.
    """)
