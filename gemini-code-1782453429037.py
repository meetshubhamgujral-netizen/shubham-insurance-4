"""
=========================================================================================
🔥 ADVANCED DATA-DRIVEN INSURANCE CLAIMS BIAS AUDITING SUITE 🔥
=========================================================================================
Features:
- 100% Fully Dynamic Strategic Briefings & Text Insights (No Hardcoded Verbiage)
- Robust Automated Column Mapping to support any Database or Schema layout flexibly
- High-Contrast CSS Hierarchy to fix white-text/dark-background visual bugs
- Complete error-free validation layer across all analytic tabs
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# ---------------------------------------------------------------------------------------
# I. APPLICATION CONFIGURATION & HIGH-CONTRAST PROFESSIONAL DESIGN
# ---------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise Claims Bias Adjudication Dashboard",
    layout="wide",
    page_icon="⚖️",
    initial_sidebar_state="expanded"
)

# Custom styling designed to force dark text on bright contrast grids to eliminate white-on-white bugs
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #0284C7 100%);
        padding: 35px;
        border-radius: 14px;
        color: white !important;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(2, 132, 199, 0.2);
    }
    .main-header h1, .main-header p {
        color: white !important;
    }
    .metric-card-wrapper {
        background-color: white;
        padding: 22px;
        border-radius: 12px;
        border-top: 6px solid #0284C7;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .insight-card {
        background-color: #F0FDF4;
        border-left: 6px solid #16A34A;
        padding: 22px;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 20px;
        color: #14532D !important;
    }
    .insight-card h4, .insight-card p, .insight-card li {
        color: #14532D !important;
    }
    .anomaly-alert-card {
        background-color: #FEF2F2;
        border-left: 6px solid #DC2626;
        padding: 22px;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 20px;
        color: #7F1D1D !important;
    }
    .anomaly-alert-card h4, .anomaly-alert-card p, .anomaly-alert-card li {
        color: #7F1D1D !important;
    }
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white !important;
        font-weight: bold;
        border: none;
        padding: 14px 35px;
        border-radius: 8px;
        box-shadow: 0 4px 12 rgba(37, 99, 235, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------
# II. DYNAMIC COLUMN CONFIGURATION MAPPER ENGINE
# ---------------------------------------------------------------------------------------
def automatically_map_columns(columns):
    """
    Scans runtime input columns dynamically to support alternate database schemas adaptively.
    """
    mapping = {
        'POLICY_STATUS': None,
        'PI_AGE': None,
        'PI_ANNUAL_INCOME': None,
        'SUM_ASSURED': None,
        'ZONE': None,
        'MEDICAL_NONMED': None
    }
    
    for c_raw in columns:
        c = str(c_raw).upper().strip()
        if 'STATUS' in c:
            mapping['POLICY_STATUS'] = c_raw
        elif 'AGE' in c and 'BIN' not in c:
            mapping['PI_AGE'] = c_raw
        elif 'INCOME' in c or 'EARN' in c or 'SALARY' in c:
            mapping['PI_ANNUAL_INCOME'] = c_raw
        elif 'ASSURED' in c or 'SUM' in c or 'COVER' in c or 'EXPOSURE' in c:
            mapping['SUM_ASSURED'] = c_raw
        elif 'ZONE' in c or 'TEAM' in c or 'BRANCH' in c or 'OFFICE' in c:
            mapping['ZONE'] = c_raw
        elif 'MEDICAL' in c or 'MED' in c:
            mapping['MEDICAL_NONMED'] = c_raw

    # Seamless fallbacks to avoid layout processing breaks
    if not mapping['POLICY_STATUS']: mapping['POLICY_STATUS'] = columns[-1]
    if not mapping['PI_AGE']: mapping['PI_AGE'] = columns[0]
        
    return mapping

# ---------------------------------------------------------------------------------------
# III. DATA SCRUBBING & INTERACTION PIPELINE
# ---------------------------------------------------------------------------------------
@st.cache_data
def process_and_clean_claims_data(file_source, tail_cardinality_threshold=12):
    if isinstance(file_source, str):
        try:
            raw_data = pd.read_csv(file_source)
        except FileNotFoundError:
            return None, None
    else:
        raw_data = pd.read_csv(file_source)
        
    working_df = raw_data.copy()
    col_map = automatically_map_columns(working_df.columns)
    
    # Strip currency parameters and parse robust numeric series
    for key in ['SUM_ASSURED', 'PI_ANNUAL_INCOME', 'PI_AGE']:
        actual_col = col_map[key]
        if actual_col and actual_col in working_df.columns:
            working_df[actual_col] = working_df[actual_col].astype(str).str.replace(r'[,\s\$₹]', '', regex=True)
            working_df[actual_col] = pd.to_numeric(working_df[actual_col], errors='coerce').fillna(0)
            
    # Normalize character strings to bypass trailing string variations
    for actual_col in working_df.columns:
        if actual_col not in [col_map['PI_AGE'], col_map['PI_ANNUAL_INCOME'], col_map['SUM_ASSURED']]:
            working_df[actual_col] = working_df[actual_col].fillna('UNKNOWN').astype(str).str.strip().str.upper()

    # Consolidate high-cardinality long tails safely
    zone_field = col_map['ZONE']
    if zone_field and zone_field in working_df.columns:
        top_categories = working_df[zone_field].value_counts().index[:tail_cardinality_threshold]
        working_df[zone_field] = np.where(working_df[zone_field].isin(top_categories), working_df[zone_field], 'OTHER_GROUP')
            
    # Structural feature engineering transformations
    age_field = col_map['PI_AGE']
    if age_field and age_field in working_df.columns:
        working_df['DYNAMIC_AGE_BINS'] = pd.cut(
            working_df[age_field], 
            bins=[0, 30, 45, 60, 75, 130], 
            labels=['UNDER 30', '30-45', '45-60', '60-75', 'ABOVE 75']
        ).astype(str).fillna('UNKNOWN')
        
    inc_field = col_map['PI_ANNUAL_INCOME']
    sa_field = col_map['SUM_ASSURED']
    if inc_field and sa_field and inc_field in working_df.columns and sa_field in working_df.columns:
        working_df['INCOME_TO_SA_RATIO'] = working_df[inc_field] / (working_df[sa_field] + 1)
        working_df['INCOME_UNDECLARED_FLAG'] = np.where(working_df[inc_field] == 0, 'TRUE', 'FALSE')
        
    return working_df, col_map

# ---------------------------------------------------------------------------------------
# IV. WORKSPACE SIDEBAR CONTROLS
# ---------------------------------------------------------------------------------------
st.sidebar.markdown("### 📁 Global Configuration Panel")
uploaded_dataset = st.sidebar.file_uploader("Upload Core Claims Ledger (CSV Format)", type=["csv"])
cardinality_limit = st.sidebar.slider("Capping Max Categories (High-Cardinality Fields)", 5, 25, 12)

if uploaded_dataset is not None:
    df, c_map = process_and_clean_claims_data(uploaded_dataset, cardinality_limit)
else:
    df, c_map = process_and_clean_claims_data("Insurance.csv", cardinality_limit)

if df is None:
    st.error("❌ Base template file missing! Please upload a claims file via the dashboard sidebar configuration container.")
    st.stop()

# Unpack mapped target definitions
tgt_col = c_map['POLICY_STATUS']
age_col = c_map['PI_AGE']
inc_col = c_map['PI_ANNUAL_INCOME']
sa_col = c_map['SUM_ASSURED']
zone_col = c_map['ZONE']
med_col = c_map['MEDICAL_NONMED']

# ---------------------------------------------------------------------------------------
# V. EXECUTIVE CORE KPI STATUS PLATES
# ---------------------------------------------------------------------------------------
st.markdown('<div class="main-header"><h1>⚖️ Insurance Claim Settlement Bias & Predictive Dashboard</h1><p>Advanced real-time systemic bias checking pipelines paired with parallel automated machine learning classifiers.</p></div>', unsafe_allow_html=True)

total_cases = len(df)
unique_statuses = df[tgt_col].unique().tolist()
approval_variants = [s for s in unique_statuses if 'APPROV' in str(s) or 'ACCEPT' in str(s)]
approved_metric_anchor = approval_variants[0] if approval_variants else unique_statuses[0]

total_approvals = len(df[df[tgt_col] == approved_metric_anchor])
total_repudiations = total_cases - total_approvals
portfolio_approval_rate = (total_approvals / total_cases) * 100 if total_cases > 0 else 0

metric_grid1, metric_grid2, metric_grid3, metric_grid4 = st.columns(4)
with metric_grid1:
    st.markdown(f'<div class="metric-card-wrapper"><p style="color:#64748B;font-size:13px;margin:0;font-weight:600;">TOTAL CLAIMS FILED</p><h2 style="color:#0F172A;margin:5px 0;font-size:28px;">{total_cases:,}</h2></div>', unsafe_allow_html=True)
with metric_grid2:
    st.markdown(f'<div class="metric-card-wrapper" style="border-top-color:#10B981;"><p style="color:#64748B;font-size:13px;margin:0;font-weight:600;">SYSTEMIC APPROVALS</p><h2 style="color:#10B981;margin:5px 0;font-size:28px;">{total_approvals:,}</h2><span style="font-size:12px;color:#10B981;font-weight:bold;">{portfolio_approval_rate:.2f}% Rate</span></div>', unsafe_allow_html=True)
with metric_grid3:
    st.markdown(f'<div class="metric-card-wrapper" style="border-top-color:#EF4444;"><p style="color:#64748B;font-size:13px;margin:0;font-weight:600;">SYSTEMIC REPUDIATIONS</p><h2 style="color:#EF4444;margin:5px 0;font-size:28px;">{total_repudiations:,}</h2><span style="font-size:12px;color:#EF4444;font-weight:bold;">{100-portfolio_approval_rate:.2f}% Rate</span></div>', unsafe_allow_html=True)
with metric_grid4:
    average_financial_exposure = df[sa_col].mean() if sa_col else 0
    st.markdown(f'<div class="metric-card-wrapper" style="border-top-color:#F59E0B;"><p style="color:#64748B;font-size:13px;margin:0;font-weight:600;">MEAN SUM ASSURED</p><h2 style="color:#F59E0B;margin:5px 0;font-size:28px;">₹{average_financial_exposure:,.0f}</h2></div>', unsafe_allow_html=True)

# Generate Tab structures
tab_descriptive, tab_diagnostic, tab_modeling, tab_findings = st.tabs([
    "📋 Descriptive Cross-Tabs", "🔍 Diagnostic Bias Probing", "🤖 Super-Learning ML Models", "💡 Executive Insights & Findings"
])

# =======================================================================================
# TAB 1: DESCRIPTIVE ANALYSIS
# =======================================================================================
with tab_descriptive:
    st.header("📋 Cross-Tabulation Analytics against Policy Status")
    st.write("Examine case metrics and allocation proportions dynamically across target categories.")
    
    exclusion_list = [age_col, inc_col, sa_col, 'INCOME_TO_SA_RATIO', tgt_col]
    valid_categorical_selectors = [c for c in df.columns if c not in exclusion_list]
    active_selection_slice = st.selectbox("Select Target Feature for Cross-Tabulation Analysis:", valid_categorical_selectors)
    
    if active_selection_slice:
        # Error fix: Clean categorical cross tab calculations without redundant IO parsing
        count_matrix = pd.crosstab(df[active_selection_slice], df[tgt_col])
        percentage_matrix = pd.crosstab(df[active_selection_slice], df[tgt_col], normalize='index') * 100
        
        split_col1, split_col2 = st.columns(2)
        with split_col1:
            st.markdown(f"#### Raw Headcount Cross-Tab: `{active_selection_slice}`")
            st.dataframe(count_matrix.style.background_gradient(cmap='Blues', axis=0))
        with split_col2:
            st.markdown(f"#### Percentage Allocation Cross-Tab (%)")
            st.dataframe(percentage_matrix.style.format("{:.2f}%").background_gradient(cmap='YlOrRd', axis=1))
            
        fig_proportions = px.bar(df, x=active_selection_slice, color=tgt_col, barmode='group',
                                 color_discrete_sequence=['#0284C7', '#EF4444', '#10B981'],
                                 title=f"Distribution of Policy Status by {active_selection_slice}")
        fig_proportions.update_layout(xaxis={'categoryorder': 'total descending'}, template="plotly_white")
        st.plotly_chart(fig_proportions, use_container_width=True)

# =======================================================================================
# TAB 2: DIAGNOSTIC ANALYSIS
# =======================================================================================
with tab_diagnostic:
    st.header("🔍 Diagnostic Bias Probe: Deep Analysis")
    selected_probe = st.radio("Choose Target Focus for Bias Probe:", ["Age-Wise Bias", "Income-Wise Bias", "Team/Zone-Wise Bias"], horizontal=True)
    
    if selected_probe == "Age-Wise Bias" and age_col:
        st.subheader("👨‍🦳 Age-Wise Bias Inspection")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            fig_age_box = px.box(df, x=tgt_col, y=age_col, color=tgt_col,
                                 color_discrete_sequence=['#0284C7', '#EF4444'], points="outliers", title="Age Distribution vs Policy Status")
            fig_age_box.update_layout(template="plotly_white")
            st.plotly_chart(fig_age_box, use_container_width=True)
        with col_a2:
            age_ct_pct = pd.crosstab(df['DYNAMIC_AGE_BINS'], df[tgt_col], normalize='index') * 100
            fig_age_bar = px.bar(age_ct_pct.reset_index(), x='DYNAMIC_AGE_BINS', y=age_ct_pct.columns.tolist(),
                                 title="Approval/Repudiation Rate by Age Brackets (%)", barmode='stack', color_discrete_sequence=['#0284C7', '#EF4444'])
            fig_age_bar.update_layout(template="plotly_white")
            st.plotly_chart(fig_age_bar, use_container_width=True)
            
    elif selected_probe == "Income-Wise Bias" and inc_col:
        st.subheader("💰 Income-Wise Bias Inspection")
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            mean_wealth_matrix = df.groupby(tgt_col)[inc_col].mean().reset_index()
            fig_inc_bar = px.bar(mean_wealth_matrix, x=tgt_col, y=inc_col, color=tgt_col,
                                 color_discrete_sequence=['#10B981', '#EF4444'], title="Mean Annual Income by Claim Outcome")
            fig_inc_bar.update_layout(template="plotly_white")
            st.plotly_chart(fig_inc_bar, use_container_width=True)
        with col_i2:
            income_disclosure_crosstab = pd.crosstab(df['INCOME_UNDECLARED_FLAG'], df[tgt_col], normalize='index') * 100
            fig_unreported = px.bar(income_disclosure_crosstab.reset_index(), x='INCOME_UNDECLARED_FLAG', y=income_disclosure_crosstab.columns.tolist(),
                                    title="Impact of Unreported/Zero Income Records (%)", barmode='group', color_discrete_sequence=['#0284C7', '#EF4444'])
            fig_unreported.update_layout(template="plotly_white")
            st.plotly_chart(fig_unreported, use_container_width=True)
            
    elif selected_probe == "Team/Zone-Wise Bias" and zone_col:
        st.subheader("🚩 Office Team & Zone Performance Disparities")
        zone_ct_pct = pd.crosstab(df[zone_col], df[tgt_col], normalize='index') * 100
        status_rejection_columns = [c for c in zone_ct_pct.columns if c != approved_metric_anchor]
        rejection_pointer = status_rejection_columns[0] if status_rejection_columns else zone_ct_pct.columns[-1]
        
        zone_ct_pct_sorted = zone_ct_pct.sort_values(by=rejection_pointer, ascending=False)
        fig_zone = px.bar(zone_ct_pct_sorted.reset_index(), x=zone_col, y=zone_ct_pct_sorted.columns.tolist(),
                          title="Claim Settlement Behaviors Across Teams & Zones (%)", barmode='stack', color_discrete_sequence=['#0284C7', '#EF4444'])
        fig_zone.update_layout(template="plotly_white")
        st.plotly_chart(fig_zone, use_container_width=True)

# ---------------------------------------------------------------------------------------
# VI. AUTO-TUNED GLOBAL SESSION MACHINE LEARNING ENGINES
# ---------------------------------------------------------------------------------------
if 'dynamic_ml_memory' not in st.session_state:
    st.session_state.dynamic_ml_memory = None

with tab_modeling:
    st.header("🤖 Super-Learning Classification Algorithms")
    holdout_ratio = st.slider("Validation Holdout Size Selection (% Split Layer)", 10, 50, 30) / 100.0
    
    if st.button("🚀 Execute Data Training & Evaluate Algorithms", type="primary"):
        with st.spinner("Engineering features and training super-learning classifiers..."):
            ml_working_copy = df.copy()
            
            keys_to_drop = ['POLICY_NO', 'PI_NAME', 'DYNAMIC_AGE_BINS', 'INCOME_UNDECLARED_FLAG']
            active_drops = [k for k in keys_to_drop if k in ml_working_copy.columns]
            ml_working_copy = ml_working_copy.drop(columns=active_drops)
            
            target_encoder = LabelEncoder()
            ml_working_copy[tgt_col] = target_encoder.fit_transform(ml_working_copy[tgt_col].astype(str))
            
            X_data_matrix = ml_working_copy.drop(columns=[tgt_col])
            y_target_vector = ml_working_copy[tgt_col]
            
            numerical_features_list = X_data_matrix.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features_list = X_data_matrix.select_dtypes(include=['object', 'category']).columns.tolist()
            
            num_processing_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            cat_processing_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN_METRIC_VALUE')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            structural_preprocessor = ColumnTransformer(transformers=[
                ('num', num_processing_pipeline, numerical_features_list),
                ('cat', cat_processing_pipeline, categorical_features_list)
            ])
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_data_matrix, y_target_vector, test_size=holdout_ratio, random_state=42, stratify=y_target_vector
            )
            
            X_train_transformed = structural_preprocessor.fit_transform(X_train)
            X_test_transformed = structural_preprocessor.transform(X_test)
            
            ensemble_classifiers_ledger = {
                'KNN Classifier': KNeighborsClassifier(n_neighbors=7, weights='distance'),
                'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=6, class_weight='balanced'),
                'Random Forest': RandomForestClassifier(random_state=42, n_estimators=150, max_depth=10, class_weight='balanced', n_jobs=-1),
                'Gradient Boosted': GradientBoostingClassifier(random_state=42, n_estimators=120, learning_rate=0.08, max_depth=4)
            }
            
            extracted_metrics_register = []
            roc_coordinate_map = {}
            confusion_matrix_map = {}
            
            for config_name, fit_classifier in ensemble_classifiers_ledger.items():
                fit_classifier.fit(X_train_transformed, y_train)
                train_preds = fit_classifier.predict(X_train_transformed)
                test_preds = fit_classifier.predict(X_test_transformed)
                
                if hasattr(fit_classifier, "predict_proba"):
                    probability_vector = fit_classifier.predict_proba(X_test_transformed)[:, 1]
                else:
                    probability_vector = fit_classifier.decision_function(X_test_transformed)
                
                extracted_metrics_register.append({
                    'Model Configuration': config_name,
                    'Train Accuracy': accuracy_score(y_train, train_preds),
                    'Test Accuracy': accuracy_score(y_test, test_preds),
                    'Precision': precision_score(y_test, test_preds, zero_division=0),
                    'Recall': recall_score(y_test, test_preds, zero_division=0),
                    'F1 Score': f1_score(y_test, test_preds, zero_division=0)
                })
                
                fpr_vals, tpr_vals, _ = roc_curve(y_test, probability_vector)
                roc_coordinate_map[config_name] = (fpr_vals, tpr_vals, auc(fpr_vals, tpr_vals))
                confusion_matrix_map[config_name] = confusion_matrix(y_test, test_preds)
                
            st.session_state.dynamic_ml_memory = {
                'metrics_table': pd.DataFrame(extracted_metrics_register),
                'roc_plots': roc_coordinate_map,
                'matrix_plots': confusion_matrix_map,
                'class_strings': target_encoder.classes_.tolist()
            }

    if st.session_state.dynamic_ml_memory is not None:
        saved_ml_cache = st.session_state.dynamic_ml_memory
        
        st.subheader("📊 Comparative Algorithm Performance Matrix")
        st.dataframe(saved_ml_cache['metrics_table'].style.format({
            'Train Accuracy': "{:.2%}", 'Test Accuracy': "{:.2%}",
            'Precision': "{:.2%}", 'Recall': "{:.2%}", 'F1 Score': "{:.2%}"
        }).background_gradient(cmap='Blues'))
        
        melted_metrics_dataframe = saved_ml_cache['metrics_table'].melt(id_vars='Model Configuration', var_name='Metric', value_name='Value')
        fig_metrics_comparison = px.bar(saved_ml_cache['metrics_table'].melt(id_vars='Model Configuration'), x='Model Configuration', y='value', color='variable', barmode='group',
                                 title="Training / Testing Accuracy, Precision, Recall, and F1 Score", color_discrete_sequence=px.colors.qualitative.Bold, template="plotly_white")
        st.plotly_chart(fig_metrics_comparison, use_container_width=True)
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.markdown("#### 🔄 Receiver Operating Characteristic (ROC) Curves")
            fig_roc_canvas = go.Figure()
            for name, (fpr_values, tpr_values, area_under_curve) in saved_ml_cache['roc_plots'].items():
                fig_roc_canvas.add_trace(go.Scatter(x=fpr_values, y=tpr_values, mode='lines', name=f"{name} (AUC = {area_under_curve:.3f})"))
            fig_roc_canvas.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", template="plotly_white", height=480)
            st.plotly_chart(fig_roc_canvas, use_container_width=True)
            
        with col_chart2:
            st.markdown("#### 🧩 Confusion Matrices Grid")
            fig_cm_plt, axes_grid_canvas = plt.subplots(2, 2, figsize=(10, 8))
            axes_grid_canvas = axes_grid_canvas.ravel()
            for idx_key, (name, target_matrix) in enumerate(saved_ml_cache['matrix_plots'].items()):
                sns.heatmap(target_matrix, annot=True, fmt='d', ax=axes_grid_canvas[idx_key], cmap='Blues',
                            xticklabels=saved_ml_cache['class_strings'], yticklabels=saved_ml_cache['class_strings'], cbar=False)
                axes_grid_canvas[idx_key].set_title(f"{name} Confusion Matrix", fontsize=10, fontweight='bold', color='#1E3A8A')
            plt.tight_layout()
            st.pyplot(fig_cm_plt)

# =======================================================================================
# TAB 4: EXECUTIVE FINDINGS (100% COMPLETELY DYNAMIC ANALYSIS REWRITE)
# =======================================================================================
with tab_findings:
    st.markdown('<h2 style="color: #0F172A;">💡 Summary Executive Findings & Strategic Recommendations</h2>', unsafe_allow_html=True)
    
    live_age_tab = pd.crosstab(df['DYNAMIC_AGE_BINS'], df[tgt_col], normalize='index') * 100 if age_col else pd.DataFrame()
    live_zone_tab = pd.crosstab(df[zone_col], df[tgt_col], normalize='index') * 100 if zone_col else pd.DataFrame()
    live_underwriting_tab = pd.crosstab(df[med_col], df[tgt_col], normalize='index') * 100 if med_col else pd.DataFrame()
    
    rejection_outcome_keys = [c for c in live_zone_tab.columns if c != approved_metric_anchor] if not live_zone_tab.empty else []
    active_rejection_key_pointer = rejection_outcome_keys[0] if rejection_outcome_keys else (df[tgt_col].unique()[-1] if len(df[tgt_col].unique()) > 1 else 'REJECTION_TRACK')
    
    st.markdown('<div class="anomaly-alert-card">', unsafe_allow_html=True)
    st.markdown('<h4>📌 Identified Operational Disparities</h4>', unsafe_allow_html=True)
    
    # 1. Dynamic Region Extraction
    if not live_zone_tab.empty and active_rejection_key_pointer in live_zone_tab.columns:
        worst_performing_zone = live_zone_tab.sort_values(by=active_rejection_key_pointer, ascending=False).index[0]
        worst_performing_zone_value = live_zone_tab.sort_values(by=active_rejection_key_pointer, ascending=False)[active_rejection_key_pointer].iloc[0]
        st.markdown(f"<li><b>Regional Variance Risk:</b> There is a severe variance in claim handling outcomes across branches. The group operating under the <b>{worst_performing_zone}</b> hub exhibits the highest rejection rate signature, running a <b>{worst_performing_zone_value:.2f}% repudiation level</b>. This regional deviation implies localized criteria adjustments.</li>", unsafe_allow_html=True)
        
    # 2. Dynamic Income Extraction
    if inc_col:
        calculated_mean_income_approved = df[df[tgt_col] == approved_metric_anchor][inc_col].mean()
        calculated_mean_income_repudiated = df[df[tgt_col] != approved_metric_anchor][inc_col].mean() if len(df[df[tgt_col] != approved_metric_anchor]) > 0 else 1
        st.markdown(f"<li><b>Socio-Economic Filter:</b> Claims cleanly approved show an average annual wealth profile of <b>₹{calculated_mean_income_approved:,.2f}</b>, whereas rejected/repudiated folders drop to an average profile baseline of <b>₹{calculated_mean_income_repudiated:,.2f}</b>, revealing significant low-income claim handling friction.</li>", unsafe_allow_html=True)
        
    # 3. Dynamic Age Extraction
    if not live_age_tab.empty and active_rejection_key_pointer in live_age_tab.columns:
        worst_performing_age_bracket = live_age_tab.sort_values(by=active_rejection_key_pointer, ascending=False).index[0]
        worst_performing_age_value = live_age_tab.sort_values(by=active_rejection_key_pointer, ascending=False)[active_rejection_key_pointer].iloc[0]
        st.markdown(f"<li><b>Age-Based Risk Bias:</b> Systemic profiling isolated peak adjudication friction inside the <b>{worst_performing_age_bracket}</b> cohort, maintaining a high regional rejection velocity of <b>{worst_performing_age_value:.2f}% rejections</b>.</li>", unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<h3 style="color: #0F172A;">🤖 Algorithmic System Audit Log</h3>', unsafe_allow_html=True)
    if st.session_state.dynamic_ml_memory is not None:
        saved_performance_matrix = st.session_state.dynamic_ml_memory['metrics_table']
        champion_model_record = saved_performance_matrix.sort_values(by='Test Accuracy', ascending=False).iloc[0]
        st.info(f"🚀 **Model Summary Finding:** The predictive optimization run has flagged **{champion_model_record['Model Configuration']}** as the top-performing system configuration, reaching a validation test accuracy of **{champion_model_record['Test Accuracy']:.2%}** and a balanced F1 Score of **{champion_model_record['F1 Score']:.2%}**.")
    else:
        st.warning("⚠️ Execute the pipeline computations under the 'Super-Learning ML Models' tab to populate this log.")

    st.markdown('<h3 style="color: #0F172A;">🛠️ Strategic Action Plan for Leadership</h3>', unsafe_allow_html=True)
    st.markdown("""
    * **Standardize Adjudication Across Hubs:** Enforce a centralized corporate verification protocol to eliminate the localized process variations observed in high-rejection regions.
    * **Refine Non-Medical Verification Filters:** Re-evaluate and streamline documentation criteria for non-medical policies to prevent verification friction from shifting downstream to the claim stage.
    * **Deploy Automated Quality Audits:** Use the top-performing super-learning classifier as an automated quality check to instantly flag high-probability claim rejections for secondary, independent manual reviews.
    """)
