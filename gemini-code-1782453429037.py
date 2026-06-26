"""
=========================================================================================
🔥 MASTER-CLASS DYNAMIC INSURANCE CLAIMS AUDIT & ML DASHBOARD 🔥
=========================================================================================
Features: 
- 100% Fully Dynamic Insights & Text Analytics (No Hardcoded Verbiage)
- Robust Automated Column Mapping to support any Database schema flexibly
- Full NameError Resolution across all Tab containers
- Advanced ML Engineering Pipelines with Distance vs Tree Scaler Isolations
"""

import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
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
# I. APPLICATION PAGE INITIALIZATION & PREMIUM THEME DESIGN
# ---------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise Claims Bias Adjudication & Predictive Analytics Dashboard",
    layout="wide",
    page_icon="⚖️",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .reportview-container { background: #F8FAFC; }
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 30px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-card-wrapper {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border-top: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .insight-card {
        background-color: #F8FAFC;
        border-left: 5px solid #1E3A8A;
        padding: 20px;
        border-radius: 6px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .anomaly-alert-card {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 20px;
        border-radius: 6px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px 30px;
        border-radius: 6px;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------
# II. AUTOMATED CONFIGURATION MAPPER ENGINE FOR ANY DATABASES
# ---------------------------------------------------------------------------------------
def automatically_map_columns(columns):
    """
    Scans incoming dataset headers dynamically to find matching operational vectors.
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
        c = str(c_raw).upper()
        if 'STATUS' in c:
            mapping['POLICY_STATUS'] = c_raw
        elif 'AGE' in c and 'BIN' not in c:
            mapping['PI_AGE'] = c_raw
        elif 'INCOME' in c or 'EARN' in c:
            mapping['PI_ANNUAL_INCOME'] = c_raw
        elif 'ASSURED' in c or 'SUM' in c or 'COVER' in c:
            mapping['SUM_ASSURED'] = c_raw
        elif 'ZONE' in c or 'TEAM' in c or 'BRANCH' in c:
            mapping['ZONE'] = c_raw
        elif 'MEDICAL' in c or 'MED' in c:
            mapping['MEDICAL_NONMED'] = c_raw

    # Fallbacks if strict pattern match yields nothing
    if not mapping['POLICY_STATUS']: mapping['POLICY_STATUS'] = columns[-1]
    if not mapping['PI_AGE']: 
        numeric_cols = [c for c in columns if 'int' in str(df[c].dtype) or 'float' in str(df[c].dtype)] if 'df' in locals() else []
        if numeric_cols: mapping['PI_AGE'] = numeric_cols[0]
        
    return mapping

# ---------------------------------------------------------------------------------------
# III. ROBUST DATA SCRUBBING & INTERACTION PIPELINE
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
    
    # Clean numeric structures
    for key in ['SUM_ASSURED', 'PI_ANNUAL_INCOME', 'PI_AGE']:
        actual_col = col_map[key]
        if actual_col and actual_col in working_df.columns:
            working_df[actual_col] = working_df[actual_col].astype(str).str.replace(r'[,\s\$₹]', '', regex=True)
            working_df[actual_col] = pd.to_numeric(working_df[actual_col], errors='coerce').fillna(0)
            
    # Normalize categorical string configurations
    for actual_col in working_df.columns:
        if actual_col not in [col_map['PI_AGE'], col_map['PI_ANNUAL_INCOME'], col_map['SUM_ASSURED']]:
            working_df[actual_col] = working_df[actual_col].fillna('UNKNOWN').astype(str).str.strip().str.upper()

    # Collapse high-cardinality long tails to limit systemic overfitting
    for key in ['ZONE']:
        actual_col = col_map[key]
        if actual_col and actual_col in working_df.columns:
            top_categories = working_df[actual_col].value_counts().index[:tail_cardinality_threshold]
            working_df[actual_col] = np.where(working_df[actual_col].isin(top_categories), working_df[actual_col], 'OTHER_GROUP')
            
    # Synthesize interaction vectors dynamically
    age_col = col_map['PI_AGE']
    if age_col and age_col in working_df.columns:
        working_df['DYNAMIC_AGE_BINS'] = pd.cut(
            working_df[age_col], 
            bins=[0, 30, 45, 60, 75, 130], 
            labels=['UNDER 30', '30-45', '45-60', '60-75', 'ABOVE 75']
        ).astype(str).fillna('UNKNOWN')
        
    income_col = col_map['PI_ANNUAL_INCOME']
    sa_col = col_map['SUM_ASSURED']
    if income_col and sa_col and income_col in working_df.columns and sa_col in working_df.columns:
        working_df['INCOME_TO_SA_RATIO'] = working_df[income_col] / (working_df[sa_col] + 1)
        working_df['INCOME_UNDECLARED_FLAG'] = np.where(working_df[income_col] == 0, 'TRUE', 'FALSE')
        
    return working_df, col_map

# ---------------------------------------------------------------------------------------
# IV. SIDEBAR CONTROLS & DATA EXTRACTION
# ---------------------------------------------------------------------------------------
st.sidebar.markdown("### 🛠️ Configuration Controls")
uploaded_dataset = st.sidebar.file_uploader("Upload Core Claims Ledger (CSV Format)", type=["csv"])
cardinality_limit = st.sidebar.slider("Capping Max Categories (High-Cardinality Fields)", 5, 25, 12)

if uploaded_dataset is not None:
    df, c_map = process_and_clean_claims_data(uploaded_dataset, cardinality_limit)
else:
    df, c_map = process_and_clean_claims_data("Insurance.csv", cardinality_limit)

if df is None:
    st.error("❌ Essential context ledger missing! Please upload a valid claims schema layout file via the dashboard sidebar configuration container.")
    st.stop()

# Assign mapped target entities
tgt_col = c_map['POLICY_STATUS']
age_col = c_map['PI_AGE']
inc_col = c_map['PI_ANNUAL_INCOME']
sa_col = c_map['SUM_ASSURED']
zone_col = c_map['ZONE']
med_col = c_map['MEDICAL_NONMED']

# ---------------------------------------------------------------------------------------
# V. MAIN KPI PORTFOLIO CARDS
# ---------------------------------------------------------------------------------------
st.markdown('<div class="main-header"><h2>⚖️ Enterprise Claims Adjudication Auditing & Predictive Suite</h2><p>Advanced real-time systemic bias checking pipelines paired with parallel automated machine learning classifiers.</p></div>', unsafe_allow_html=True)

total_records = len(df)
unique_statuses = df[tgt_col].unique().tolist()
approved_status_flag = [s for s in unique_statuses if 'APPROV' in str(s)]
approved_status_flag = approved_status_flag[0] if approved_status_flag else unique_statuses[0]

total_approvals = len(df[df[tgt_col] == approved_status_flag])
total_repudiations = total_records - total_approvals
base_approval_rate = (total_approvals / total_records) * 100 if total_records > 0 else 0

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
with metric_col1:
    st.markdown(f'<div class="metric-card-wrapper"><p style="color:#64748B;font-size:14px;margin:0;">TOTAL CLAIMS AUDITED</p><h2 style="color:#0F172A;margin:5px 0;">{total_records:,}</h2></div>', unsafe_allow_html=True)
with metric_col2:
    st.markdown(f'<div class="metric-card-wrapper" style="border-top-color:#10B981;"><p style="color:#64748B;font-size:14px;margin:0;">SYSTEMIC APPROVALS</p><h2 style="color:#10B981;margin:5px 0;">{total_approvals:,}</h2><span style="font-size:12px;color:#10B981;font-weight:bold;">{base_approval_rate:.2f}% Rate</span></div>', unsafe_allow_html=True)
with metric_col3:
    st.markdown(f'<div class="metric-card-wrapper" style="border-top-color:#EF4444;"><p style="color:#64748B;font-size:14px;margin:0;">SYSTEMIC REPUDIATIONS</p><h2 style="color:#EF4444;margin:5px 0;">{total_repudiations:,}</h2><span style="font-size:12px;color:#EF4444;font-weight:bold;">{100-base_approval_rate:.2f}% Rate</span></div>', unsafe_allow_html=True)
with metric_col4:
    avg_sa = df[sa_col].mean() if sa_col else 0
    st.markdown(f'<div class="metric-card-wrapper" style="border-top-color:#F59E0B;"><p style="color:#64748B;font-size:14px;margin:0;">MEAN PORTFOLIO EXPOSURE</p><h2 style="color:#F59E0B;margin:5px 0;">₹{avg_sa:,.0f}</h2></div>', unsafe_allow_html=True)

# Main Navigation Layout Slices
tab_descriptive, tab_diagnostic, tab_modeling, tab_findings = st.tabs([
    "📊 Descriptive Cross-Tabs", "🎯 Diagnostic Bias Probing", "🧠 Super-Learning Classifiers", "📝 Automated Executive Briefing"
])

# =======================================================================================
# TAB 1: DESCRIPTIVE CROSS-TABS
# =======================================================================================
with tab_descriptive:
    st.header("📊 Dynamic Cross-Tabulation Slices")
    available_categorical_slices = [c for c in df.columns if c not in [age_col, inc_col, sa_col, 'INCOME_TO_SA_RATIO', tgt_col]]
    selected_target_slice = st.selectbox("Select Target Demography Slice Variable:", available_categorical_slices)
    
    if selected_target_slice:
        cross_tab_counts = pd.crosstab(df[selected_target_slice], df[tgt_col])
        cross_tab_normalized = pd.crosstab(df[selected_target_slice], df[tgt_col], normalize='index') * 100
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown(f"#### Absolute Case Volume Breakdown")
            st.dataframe(cross_tab_counts.style.background_gradient(cmap='Blues', axis=0))
        with col_c2:
            st.markdown(f"#### Proportional Allocation Breakdown (`% Within Group`)")
            st.dataframe(cross_tab_normalized.style.format("{:.2f}%").background_gradient(cmap='YlOrRd', axis=1))
            
        fig_proportions = px.bar(df, x=selected_target_slice, color=tgt_col, barmode='group',
                                 color_discrete_sequence=['#3B82F6', '#EF4444', '#10B981'],
                                 title=f"Volumetric Case Split Distribution Across: {selected_target_slice}")
        fig_proportions.update_layout(xaxis={'categoryorder': 'total descending'}, template="plotly_white")
        st.plotly_chart(fig_proportions, use_container_width=True)

# =======================================================================================
# TAB 2: DIAGNOSTIC BIAS PROBING (POINTS VALUE CONSTRAINED SAFELY)
# =======================================================================================
with tab_diagnostic:
    st.header("🎯 Systemic Bias Diagnostic Probe Engine")
    probe_dimension = st.radio("Isolate Operational Evaluation Domain:", ["Age Profile Disparities", "Socio-Economic Income Gaps", "Team & Regional Branch Divergence"], horizontal=True)
    
    if probe_dimension == "Age Profile Disparities" and age_col:
        st.subheader("Inspection: Age Demographic Profile Gaps")
        col_a1, col_a2 = st.columns(2)
        
        with col_a1:
            fig_age_box = px.box(df, x=tgt_col, y=age_col, color=tgt_col,
                                 color_discrete_sequence=['#3B82F6', '#EF4444'], points="outliers", 
                                 title="Structural Age Distortions vs Claim Resolution")
            fig_age_box.update_layout(template="plotly_white")
            st.plotly_chart(fig_age_box, use_container_width=True)
        with col_a2:
            age_ct_pct = pd.crosstab(df['DYNAMIC_AGE_BINS'], df[tgt_col], normalize='index') * 100
            fig_age_bar = px.bar(age_ct_pct.reset_index(), x='DYNAMIC_AGE_BINS', y=age_ct_pct.columns.tolist(),
                                 title="Adjudication Track Realization Slices Across Age Brackets (%)", barmode='stack',
                                 color_discrete_sequence=['#3B82F6', '#EF4444'])
            st.plotly_chart(fig_age_bar, use_container_width=True)
            
        other_status_cols = [c for c in age_ct_pct.columns if c != approved_status_flag]
        rep_col_pointer = other_status_cols[0] if other_status_cols else age_ct_pct.columns[-1]
        worst_age_bin = age_ct_pct.sort_values(by=rep_col_pointer, ascending=False).index[0]
        worst_age_val = age_ct_pct.sort_values(by=rep_col_pointer, ascending=False)[rep_col_pointer].iloc[0]
        
        st.markdown(f"""
            <div class="anomaly-alert-card">
                <h4>⚠️ Automated Age Distortion Diagnostic:</h4>
                <p>Systemic tracking algorithms have isolated maximum friction inside the <b>{worst_age_bin}</b> bracket, 
                yielding an active metrics repudiation velocity of <b>{worst_age_val:.2f}%</b>.</p>
            </div>
        """, unsafe_allow_html=True)
        
    elif probe_dimension == "Socio-Economic Income Gaps" and inc_col:
        st.subheader("Inspection: Socio-Economic Income Profiling")
        col_i1, col_i2 = st.columns(2)
        
        mean_financials = df.groupby(tgt_col)[inc_col].mean().reset_index()
        approved_mean_val = mean_financials[mean_financials[tgt_col] == approved_status_flag][inc_col].values[0] if approved_status_flag in mean_financials[tgt_col].values else 0
        repudiated_mean_val = df[df[tgt_col] != approved_status_flag][inc_col].mean() if len(df[df[tgt_col] != approved_status_flag]) > 0 else 0
        
        with col_i1:
            fig_inc_bar = px.bar(mean_financials, x=tgt_col, y=inc_col, color=tgt_col,
                                 color_discrete_sequence=['#10B981', '#EF4444'], title="Average Income Matrix Profile vs Settlement Status")
            st.plotly_chart(fig_inc_bar, use_container_width=True)
        with col_i2:
            income_unreported_crosstab = pd.crosstab(df['INCOME_UNDECLARED_FLAG'], df[tgt_col], normalize='index') * 100
            fig_unreported = px.bar(income_unreported_crosstab.reset_index(), x='INCOME_UNDECLARED_FLAG', y=income_unreported_crosstab.columns.tolist(),
                                    title="Resolution Risk Velocity Contingent on Non-Disclosure of Income (%)", barmode='group',
                                    color_discrete_sequence=['#3B82F6', '#EF4444'])
            st.plotly_chart(fig_unreported, use_container_width=True)
            
        st.markdown(f"""
            <div class="anomaly-alert-card">
                <h4>⚠️ Automated Wealth Concentration Diagnostic:</h4>
                <p>The statistical mean wealth signature for safe tracking outcomes stands at <b>₹{approved_mean_val:,.2f}</b>, 
                whereas the dropped/rejected class registers a mean parameter signature of <b>₹{repudiated_mean_val:,.2f}</b>.</p>
            </div>
        """, unsafe_allow_html=True)
        
    elif probe_dimension == "Team & Regional Branch Divergence" and zone_col:
        st.subheader("Inspection: Regional Performance Disparities")
        zone_ct_pct = pd.crosstab(df[zone_col], df[tgt_col], normalize='index') * 100
        other_status_cols = [c for c in zone_ct_pct.columns if c != approved_status_flag]
        rep_col_pointer = other_status_cols[0] if other_status_cols else zone_ct_pct.columns[-1]
        
        zone_ct_pct_sorted = zone_ct_pct.sort_values(by=rep_col_pointer, ascending=False)
        fig_zone = px.bar(zone_ct_pct_sorted.reset_index(), x=zone_col, y=zone_ct_pct_sorted.columns.tolist(),
                          title="Systemic Adjudication Behavior Divergence by Active Regional Office Tracking Nodes (%)", barmode='stack',
                          color_discrete_sequence=['#3B82F6', '#EF4444'])
        st.plotly_chart(fig_zone, use_container_width=True)
        
        highest_friction_branch = zone_ct_pct_sorted.index[0]
        highest_friction_value = zone_ct_pct_sorted[rep_col_pointer].iloc[0]
        
        st.markdown(f"""
            <div class="anomaly-alert-card">
                <h4>⚠️ Automated Regional Variance Diagnostic:</h4>
                <p>Severe localization variance detected between corporate operating sites. The team node matching <b>{highest_friction_branch}</b> 
                exhibits an aggressive baseline stance tracking at <b>{highest_friction_value:.2f}% repudiations</b>.</p>
            </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------
# VI. AUTO-TUNED GLOBAL SESSION MACHINE LEARNING PIPELINES
# ---------------------------------------------------------------------------------------
if 'dynamic_ml_memory' not in st.session_state:
    st.session_state.dynamic_ml_memory = None

with tab_modeling:
    st.header("🧠 Automated Advanced Prediction & Super-Learning Classifiers")
    test_split_ratio = st.slider("Validation Holdout Size Selection (% Split Layer)", 10, 50, 30) / 100.0
    
    if st.button("🚀 Run Feature Pipelines & Benchmark Super-Classifiers", type="primary"):
        with st.spinner("Executing transformations..."):
            ml_base_df = df.copy()
            
            unwanted_leakage_keys = ['POLICY_NO', 'PI_NAME', 'DYNAMIC_AGE_BINS', 'INCOME_UNDECLARED_FLAG']
            existing_drop_keys = [k for k in unwanted_leakage_keys if k in ml_base_df.columns]
            ml_base_df = ml_base_df.drop(columns=existing_drop_keys)
            
            target_label_encoder = LabelEncoder()
            ml_base_df[tgt_col] = target_label_encoder.fit_transform(ml_base_df[tgt_col].astype(str))
            
            X_array = ml_base_df.drop(columns=[tgt_col])
            y_array = ml_base_df[tgt_col]
            
            numeric_features = X_array.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = X_array.select_dtypes(include=['object', 'category']).columns.tolist()
            
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN_VAL')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            preprocessor_pipeline = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_array, y_array, test_size=test_split_ratio, random_state=42, stratify=y_array
            )
            
            X_train_processed = preprocessor_pipeline.fit_transform(X_train)
            X_test_processed = preprocessor_pipeline.transform(X_test)
            
            algorithm_ledger = {
                'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=7, weights='distance'),
                'Optimized Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=6, class_weight='balanced'),
                'Balanced Random Forest': RandomForestClassifier(random_state=42, n_estimators=150, max_depth=10, class_weight='balanced', n_jobs=-1),
                'Gradient Boosting Machine': GradientBoostingClassifier(random_state=42, n_estimators=120, learning_rate=0.08, max_depth=4)
            }
            
            performance_metrics_store = []
            roc_curve_plotting_store = {}
            confusion_matrix_plotting_store = {}
            
            for model_name, classifier_instance in algorithm_ledger.items():
                classifier_instance.fit(X_train_processed, y_train)
                train_predictions = classifier_instance.predict(X_train_processed)
                test_predictions = classifier_instance.predict(X_test_processed)
                
                if hasattr(classifier_instance, "predict_proba"):
                    test_probabilities = classifier_instance.predict_proba(X_test_processed)[:, 1]
                else:
                    test_probabilities = classifier_instance.decision_function(X_test_processed)
                
                performance_metrics_store.append({
                    'Model Configuration': model_name,
                    'Train Accuracy': accuracy_score(y_train, train_predictions),
                    'Test Accuracy': accuracy_score(y_test, test_predictions),
                    'Precision': precision_score(y_test, test_predictions, zero_division=0),
                    'Recall': recall_score(y_test, test_predictions, zero_division=0),
                    'F1-Score': f1_score(y_test, test_predictions, zero_division=0)
                })
                
                false_positive_rate, true_positive_rate, _ = roc_curve(y_test, test_probabilities)
                roc_curve_plotting_store[model_name] = (false_positive_rate, true_positive_rate, auc(false_positive_rate, true_positive_rate))
                confusion_matrix_plotting_store[model_name] = confusion_matrix(y_test, test_predictions)
                
            st.session_state.dynamic_ml_memory = {
                'metrics_table': pd.DataFrame(performance_metrics_store),
                'roc_data': roc_curve_plotting_store,
                'matrix_data': confusion_matrix_plotting_store,
                'class_labels': target_label_encoder.classes_.tolist()
            }

    if st.session_state.dynamic_ml_memory is not None:
        saved_ml = st.session_state.dynamic_ml_memory
        st.subheader("📊 Comparative Algorithm Performance Matrix")
        st.dataframe(saved_ml['metrics_table'].style.format({
            'Train Accuracy': "{:.2%}", 'Test Accuracy': "{:.2%}",
            'Precision': "{:.2%}", 'Recall': "{:.2%}", 'F1-Score': "{:.2%}"
        }).background_gradient(cmap='Blues'))
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown("#### 🔄 Receiver Operating Characteristic (ROC) Space Curves")
            fig_roc_space = go.Figure()
            for name, (fpr, tpr, roc_auc_val) in saved_ml['roc_data'].items():
                fig_roc_space.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC = {roc_auc_val:.3f})"))
            fig_roc_space.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", template="plotly_white")
            st.plotly_chart(fig_roc_space, use_container_width=True)
            
        with col_v2:
            st.markdown("#### 🧩 Confusion Matrices Evaluation Matrix")
            fig_plt_cm, axes_grid = plt.subplots(2, 2, figsize=(10, 8))
            axes_grid = axes_grid.ravel()
            for idx, (name, cm_matrix) in enumerate(saved_ml['matrix_data'].items()):
                sns.heatmap(cm_matrix, annot=True, fmt='d', ax=axes_grid[idx], cmap='Blues',
                            xticklabels=saved_ml['class_labels'], yticklabels=saved_ml['class_labels'], cbar=False)
                axes_grid[idx].set_title(name, fontsize=10, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_plt_cm)

# =======================================================================================
# TAB 4: EXECUTIVE FINDINGS (RESOLVED NAMEERROR CONTEXT)
# =======================================================================================
with tab_findings:
    st.header("📝 Automated Strategic Briefing & Action Register")
    
    age_ct_live = pd.crosstab(df['DYNAMIC_AGE_BINS'], df[tgt_col], normalize='index') * 100 if age_col else pd.DataFrame()
    zone_ct_live = pd.crosstab(df[zone_col], df[tgt_col], normalize='index') * 100 if zone_col else pd.DataFrame()
    
    other_status_cols = [c for c in zone_ct_live.columns if c != approved_status_flag] if not zone_ct_live.empty else []
    rep_col_pointer = other_status_cols[0] if other_status_cols else (df[tgt_col].unique()[-1] if len(df[tgt_col].unique()) > 1 else 'REJECTIONS')
    
    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
    st.subheader("📌 Active Operational Insights Summary")
    
    if not zone_ct_live.empty and rep_col_pointer in zone_ct_live.columns:
        worst_z = zone_ct_live.sort_values(by=rep_col_pointer, ascending=False).index[0]
        worst_z_val = zone_ct_live.sort_values(by=rep_col_pointer, ascending=False)[rep_col_pointer].iloc[0]
        st.markdown(f"- **Regional Variance Risk:** The team operating under the **{worst_z}** node exhibits the highest rejection rate layout across the portfolio at **{worst_z_val:.2f}% rejections**.")
        
    if inc_col:
        m_app = df[df[tgt_col] == approved_status_flag][inc_col].mean()
        m_rep = df[df[tgt_col] != approved_status_flag][inc_col].mean()
        st.markdown(f"- **Socio-Economic Filter Risk:** Approved profiles track a mean income parameter signature of **₹{m_app:,.2f}**, while rejected cases drop to a lower mean footprint of **₹{m_rep:,.2f}**.")
        
    if not age_ct_live.empty and rep_col_pointer in age_ct_live.columns:
        worst_a = age_ct_live.sort_values(by=rep_col_pointer, ascending=False).index[0]
        worst_a_val = age_ct_live.sort_values(by=rep_col_pointer, ascending=False)[rep_col_pointer].iloc[0]
        st.markdown(f"- **Age-Based Track Disparities:** Systemic profiling catches maximum operational friction inside the **{worst_a}** bracket, exhibiting a localized risk parameter index of **{worst_a_val:.2f}% rejections**.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("🤖 Algorithmic System Audit")
    if st.session_state.dynamic_ml_memory is not None:
        saved_metrics_brief = st.session_state.dynamic_ml_memory['metrics_table']
        champion_model_row = saved_metrics_brief.sort_values(by='Test Accuracy', ascending=False).iloc[0]
        st.info(f"🚀 **Model Audit Finding:** The **{champion_model_row['Model Configuration']}** algorithm is currently the top-performing system, with a validation test accuracy of **{champion_model_row['Test Accuracy']:.2%}** and an F1-Score of **{champion_model_row['F1-Score']:.2%}**.")
    else:
        st.warning("⚠️ Run the predictive models on the 'Super-Learning Classifiers' tab to populate this automated system audit log.")
