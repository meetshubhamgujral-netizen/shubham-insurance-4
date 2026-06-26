
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

# Custom premium styling matrix via standard markdown blocks
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
# II. ROBUST DATA SCRUBBING & INTERACTION PIPELINE
# ---------------------------------------------------------------------------------------
@st.cache_data
def process_and_clean_claims_data(file_source, tail_cardinality_threshold=10):
    """
    Cleans structural numeric strings, normalizes case syntax variations, 
    isolates zero entries as structural markers, and caps out severe long tails.
    """
    if isinstance(file_source, str):
        try:
            raw_data = pd.read_csv(file_source)
        except FileNotFoundError:
            return None
    else:
        raw_data = pd.read_csv(file_source)
        
    working_df = raw_data.copy()
    
    # Strip formatting structures and push to numeric arrays
    for financial_col in ['SUM_ASSURED', 'PI_ANNUAL_INCOME']:
        if financial_col in working_df.columns:
            working_df[financial_col] = working_df[financial_col].astype(str).str.replace(r'[,\s]', '', regex=True)
            working_df[financial_col] = pd.to_numeric(working_df[financial_col], errors='coerce').fillna(0)
            
    # Resolve structural textual inconsistencies and handle missing data allocations
    text_imputation_map = {'PI_OCCUPATION': 'Unknown', 'REASON_FOR_CLAIM': 'Not Disclosed', 'ZONE': 'UNKNOWN'}
    for text_col, fill_val in text_imputation_map.items():
        if text_col in working_df.columns:
            working_df[text_col] = working_df[text_col].fillna(fill_val).astype(str).str.strip().str.upper()

    # Address casing anomalies across structural elements (e.g., South vs SOUTH)
    categorical_targets = ['PI_GENDER', 'PAYMENT_MODE', 'EARLY_NON', 'MEDICAL_NONMED', 'PI_STATE', 'POLICY_STATUS']
    for cat_col in categorical_targets:
        if cat_col in working_df.columns:
            working_df[cat_col] = working_df[cat_col].astype(str).str.strip().str.upper()

    # Isolate missing or undeclared income entities
    if 'PI_ANNUAL_INCOME' in working_df.columns:
        working_df['INCOME_UNDECLARED_FLAG'] = np.where(working_df['PI_ANNUAL_INCOME'] == 0, 'TRUE', 'FALSE')

    # Mitigate overfitting by collapsing long-tail configurations into an 'OTHER' bin
    high_card_fields = ['ZONE', 'PI_STATE', 'PI_OCCUPATION', 'REASON_FOR_CLAIM']
    for field in high_card_fields:
        if field in working_df.columns:
            top_categories = working_df[field].value_counts().index[:tail_cardinality_threshold]
            working_df[field] = np.where(working_df[field].isin(top_categories), working_df[field], 'OTHER_GROUP')
            
    # Automated generation of derivative features for model optimization
    if 'PI_AGE' in working_df.columns:
        working_df['AGE_BINS'] = pd.cut(
            working_df['PI_AGE'], 
            bins=[0, 30, 45, 60, 75, 120], 
            labels=['UNDER 30', '30-45', '45-60', '60-75', 'ABOVE 75']
        ).astype(str)
        
    if 'PI_ANNUAL_INCOME' in working_df.columns and 'SUM_ASSURED' in working_df.columns:
        working_df['INCOME_TO_LEVERAGE_RATIO'] = working_df['PI_ANNUAL_INCOME'] / (working_df['SUM_ASSURED'] + 1)
        
    return working_df

# ---------------------------------------------------------------------------------------
# III. SIDEBAR CONTROL CONTROLLERS
# ---------------------------------------------------------------------------------------
st.sidebar.markdown("### 🛠️ Configuration Controls")
uploaded_dataset = st.sidebar.file_uploader("Upload Core Claims Ledger (CSV Format)", type=["csv"])
cardinality_limit = st.sidebar.slider("Capping Max Categories (High-Cardinality Fields)", 5, 25, 12)

# Load target matrix structure
if uploaded_dataset is not None:
    df = process_and_clean_claims_data(uploaded_dataset, cardinality_limit)
else:
    df = process_and_clean_claims_data("Insurance.csv", cardinality_limit)

if df is None:
    st.error("❌ Essential context ledger missing! Please upload a valid claims schema layout file via the dashboard sidebar configuration container.")
    st.stop()

# Validate status configurations
if 'POLICY_STATUS' not in df.columns:
    st.error("❌ Invalid target structural design! Data must hold a target column identified strictly as 'POLICY_STATUS'.")
    st.stop()

# ---------------------------------------------------------------------------------------
# IV. CORE KPI CONTAINER SUMMARY MATRICES
# ---------------------------------------------------------------------------------------
st.markdown('<div class="main-header"><h2>⚖️ Enterprise Claims Adjudication Auditing & Predictive Suite</h2><p>Advanced real-time systemic bias checking pipelines paired with parallel automated machine learning classifiers.</p></div>', unsafe_allow_html=True)

total_records = len(df)
all_statuses = df['POLICY_STATUS'].unique().tolist()
primary_approved_str = [s for s in all_statuses if 'APPROV' in s]
primary_approved_str = primary_approved_str[0] if primary_approved_str else (all_statuses[0] if all_statuses else "APPROVED")

total_approvals = len(df[df['POLICY_STATUS'] == primary_approved_str])
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
    avg_exposure = df['SUM_ASSURED'].mean() if 'SUM_ASSURED' in df.columns else 0
    st.markdown(f'<div class="metric-card-wrapper" style="border-top-color:#F59E0B;"><p style="color:#64748B;font-size:14px;margin:0;">MEAN PORTFOLIO EXPOSURE</p><h2 style="color:#F59E0B;margin:5px 0;">₹{avg_exposure:,.0f}</h2></div>', unsafe_allow_html=True)

# Generate Tab structures
tab_descriptive, tab_diagnostic, tab_modeling, tab_findings = st.tabs([
    "📊 Descriptive Cross-Tabs", 
    "🎯 Diagnostic Bias Probing", 
    "🧠 Super-Learning Classifiers", 
    "📝 Automated Executive Briefing"
])

# =======================================================================================
# TAB 1: DESCRIPTIVE CROSS-TABS
# =======================================================================================
with tab_descriptive:
    st.header("📊 Dynamic Cross-Tabulation Slices")
    st.write("Examine proportional distributions across all fields against `POLICY_STATUS` to instantly detect anomalies in volume allocation.")
    
    available_categorical_slices = [c for c in ['PI_GENDER', 'ZONE', 'PAYMENT_MODE', 'EARLY_NON', 'MEDICAL_NONMED', 'PI_STATE', 'AGE_BINS', 'INCOME_UNDECLARED_FLAG'] if c in df.columns]
    selected_target_slice = st.selectbox("Select Target Demography Slice Variable:", available_categorical_slices)
    
    if selected_target_slice:
        cross_tab_counts = pd.crosstab(df[selected_target_slice], df['POLICY_STATUS'])
        cross_tab_normalized = pd.crosstab(df[selected_target_slice], df['POLICY_STATUS'], normalize='index') * 100
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown(f"#### Absolute Case Volume Breakdown: `{selected_target_slice}`")
            st.dataframe(cross_tab_counts.style.background_gradient(cmap='Blues', axis=0))
        with col_c2:
            st.markdown(f"#### Proportional Allocation Breakdown: `% Within Category`")
            st.dataframe(cross_tab_normalized.style.format("{:.2f}%").background_gradient(cmap='YlOrRd', axis=1))
            
        st.markdown("#### Visual Distribution Analytics Plot")
        fig_proportions = px.bar(
            df, 
            x=selected_target_slice, 
            color='POLICY_STATUS',
            barmode='group',
            color_discrete_sequence=['#3B82F6', '#EF4444', '#10B981', '#F59E0B'],
            title=f"Volumetric Case Split Distribution Across Class Variable: {selected_target_slice}"
        )
        fig_proportions.update_layout(xaxis={'categoryorder': 'total descending'}, template="plotly_white")
        st.plotly_chart(fig_proportions, use_container_width=True)

# =======================================================================================
# TAB 2: DIAGNOSTIC BIAS PROBING (100% DYNAMIC GENERATION)
# =======================================================================================
with tab_diagnostic:
    st.header("🎯 Systemic Bias Diagnostic Probe Engine")
    st.write("Isolate variables dynamically to analyze operational patterns or deviations in adjudication outcomes.")
    
    probe_dimension = st.radio("Isolate Operational Evaluation Domain:", ["Age Profile Disparities", "Socio-Economic Income Gaps", "Team & Regional Branch Divergence"], horizontal=True)
    
    if probe_dimension == "Age Profile Disparities" and 'PI_AGE' in df.columns:
        st.subheader("Inspection: Age Demographic Profile Gaps")
        col_a1, col_a2 = st.columns(2)
        
        with col_a1:
            fig_age_box = px.box(df, x='POLICY_STATUS', y='PI_AGE', color='POLICY_STATUS',
                                 color_discrete_sequence=['#3B82F6', '#EF4444'], points="outer", title="Structural Age Distortions vs Claim Resolution")
            fig_age_box.update_layout(template="plotly_white")
            st.plotly_chart(fig_age_box, use_container_width=True)
        with col_a2:
            age_ct_pct = pd.crosstab(df['AGE_BINS'], df['POLICY_STATUS'], normalize='index') * 100
            fig_age_bar = px.bar(age_ct_pct.reset_index(), x='AGE_BINS', y=age_ct_pct.columns.tolist(),
                                 title="Adjudication Track Realization Slices Across Age Brackets (%)", barmode='stack',
                                 color_discrete_sequence=['#3B82F6', '#EF4444'])
            fig_age_bar.update_layout(template="plotly_white")
            st.plotly_chart(fig_age_bar, use_container_width=True)
            
        # DYNAMIC TEXT EXTRACTION
        max_rep_age_cohort = age_ct_pct.index[0]
        max_rep_age_val = 0
        all_rep_cols = [c for c in age_ct_pct.columns if 'REPUD' in str(c) or 'DENI' in str(c) or 'REJ' in str(c)]
        target_rep_col = all_rep_cols[0] if all_rep_cols else age_ct_pct.columns[-1]
        
        for idx in age_ct_pct.index:
            if age_ct_pct.loc[idx, target_rep_col] > max_rep_age_val:
                max_rep_age_val = age_ct_pct.loc[idx, target_rep_col]
                max_rep_age_cohort = idx
                
        st.markdown(f"""
            <div class="anomaly-alert-card">
                <h4>⚠️ Automated Age Distortion Diagnostic:</h4>
                <p>Systemic tracking algorithms have isolated maximum friction inside the <b>{max_rep_age_cohort}</b> bracket, 
                yielding an active repudiation index velocity of <b>{max_rep_age_val:.2f}%</b>. This deviation requires case-file review 
                to see if older or younger cohorts face unequal hurdles during document validation.</p>
            </div>
        """, unsafe_allow_html=True)
        
    elif probe_dimension == "Socio-Economic Income Gaps" and 'PI_ANNUAL_INCOME' in df.columns:
        st.subheader("Inspection: Socio-Economic Income Profiling")
        col_i1, col_i2 = st.columns(2)
        
        mean_financials = df.groupby('POLICY_STATUS')['PI_ANNUAL_INCOME'].mean().reset_index()
        approved_mean_val = mean_financials[mean_financials['POLICY_STATUS'] == primary_approved_str]['PI_ANNUAL_INCOME'].values[0] if primary_approved_str in mean_financials['POLICY_STATUS'].values else 1
        repudiated_mean_val = df[df['POLICY_STATUS'] != primary_approved_str]['PI_ANNUAL_INCOME'].mean() if len(df[df['POLICY_STATUS'] != primary_approved_str]) > 0 else 1
        income_disparity_multiple = approved_mean_val / repudiated_mean_val if repudiated_mean_val > 0 else 1
        
        with col_i1:
            fig_inc_bar = px.bar(mean_financials, x='POLICY_STATUS', y='PI_ANNUAL_INCOME', color='POLICY_STATUS',
                                 color_discrete_sequence=['#10B981', '#EF4444'], title="Portfolio Average Annual Wealth Levels by Ultimate Claim Resolution State")
            fig_inc_bar.update_layout(template="plotly_white")
            st.plotly_chart(fig_inc_bar, use_container_width=True)
        with col_i2:
            income_unreported_crosstab = pd.crosstab(df['INCOME_UNDECLARED_FLAG'], df['POLICY_STATUS'], normalize='index') * 100
            fig_unreported = px.bar(income_unreported_crosstab.reset_index(), x='INCOME_UNDECLARED_FLAG', y=income_unreported_crosstab.columns.tolist(),
                                    title="Resolution Risk Velocity Contingent on Non-Disclosure of Income (%)", barmode='group',
                                    color_discrete_sequence=['#3B82F6', '#EF4444'])
            fig_unreported.update_layout(template="plotly_white")
            st.plotly_chart(fig_unreported, use_container_width=True)
            
        st.markdown(f"""
            <div class="anomaly-alert-card">
                <h4>⚠️ Automated Wealth Concentration Diagnostic:</h4>
                <p>The statistical mean wealth factor for a safe tracking outcome stands at <b>₹{approved_mean_val:,.2f}</b>, 
                whereas the dropped/rejected class registers a mean parameter signature of <b>₹{repudiated_mean_val:,.2f}</b>. 
                Approved individuals have a wealth factor signature roughly <b>{income_disparity_multiple:.2f}x</b> larger than rejected individuals. 
                This strongly suggests lower-income applicants face greater verification hurdles.</p>
            </div>
        """, unsafe_allow_html=True)
        
    elif probe_dimension == "Team & Regional Branch Divergence" and 'ZONE' in df.columns:
        st.subheader("Inspection: Regional Performance Disparities")
        zone_ct_pct = pd.crosstab(df['ZONE'], df['POLICY_STATUS'], normalize='index') * 100
        
        all_rep_cols = [c for c in zone_ct_pct.columns if 'REPUD' in str(c) or 'DENI' in str(c) or 'REJ' in str(c)]
        target_rep_col = all_rep_cols[0] if all_rep_cols else zone_ct_pct.columns[-1]
        
        zone_ct_pct_sorted = zone_ct_pct.sort_values(by=target_rep_col, ascending=False)
        
        fig_zone = px.bar(zone_ct_pct_sorted.reset_index(), x='ZONE', y=zone_ct_pct_sorted.columns.tolist(),
                          title="Systemic Adjudication Behavior Divergence by Active Regional Office Tracking Nodes (%)", barmode='stack',
                          color_discrete_sequence=['#3B82F6', '#EF4444', '#10B981'], height=550)
        fig_zone.update_layout(template="plotly_white")
        st.plotly_chart(fig_zone, use_container_width=True)
        
        highest_friction_branch = zone_ct_pct_sorted.index[0]
        highest_friction_value = zone_ct_pct_sorted[target_rep_col].iloc[0]
        lowest_friction_branch = zone_ct_pct_sorted.index[-1]
        lowest_friction_value = zone_ct_pct_sorted[target_rep_col].iloc[-1]
        
        st.markdown(f"""
            <div class="anomaly-alert-card">
                <h4>⚠️ Automated Regional Variance Diagnostic:</h4>
                <p>Severe localization variance detected between corporate operating sites. The team node matching <b>{highest_friction_branch}</b> 
                exhibits an aggressive repudiation posture at <b>{highest_friction_value:.2f}%</b>, whereas the branch operating at 
                <b>{lowest_friction_branch}</b> tracks a low repudiation baseline of just <b>{lowest_friction_value:.2f}%</b>. This performance gap indicates 
                that claims processing rules may be inconsistent across regional offices.</p>
            </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------
# V. AUTO-TUNED GLOBAL SESSION MACHINE LEARNING CONSTRUCTS
# ---------------------------------------------------------------------------------------
if 'dynamic_ml_memory' not in st.session_state:
    st.session_state.dynamic_ml_memory = None

# =======================================================================================
# TAB 3: SUPER-LEARNING CLASSIFIERS
# =======================================================================================
with tab3:
    st.header("🧠 Automated Advanced Prediction & Super-Learning Classifiers")
    st.write("Trigger automated high-performance pipelines to engineer features, correct structural skew, and train machine learning models.")
    
    test_split_ratio = st.slider("Validation Holdout Size Selection (% Split Layer)", 10, 50, 30) / 100.0
    
    if st.button("🚀 Run Feature Pipelines & Benchmark Super-Classifiers", type="primary"):
        with st.spinner("Executing transformations, optimizing arrays, and evaluating baseline performance..."):
            
            # Recompute transformations safely on clean data arrays
            ml_base_df = df.copy()
            
            # Remove direct keys or primary indicators to protect generalization layers
            unwanted_leakage_keys = ['POLICY_NO', 'PI_NAME', 'AGE_BINS', 'INCOME_UNDECLARED_FLAG']
            existing_drop_keys = [k for k in unwanted_leakage_keys if k in ml_base_df.columns]
            ml_base_df = ml_base_df.drop(columns=existing_drop_keys)
            
            # Define explicit target mappings
            target_label_encoder = LabelEncoder()
            ml_base_df['POLICY_STATUS'] = target_label_encoder.fit_transform(ml_base_df['POLICY_STATUS'].astype(str))
            
            # Track target identification array indexes
            positive_class_index = 1 
            
            X_array = ml_base_df.drop(columns=['POLICY_STATUS'])
            y_array = ml_base_df['POLICY_STATUS']
            
            # Separate numeric and categorical pipelines
            numeric_features = X_array.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = X_array.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Construct standard pipeline components
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
            
            # Split array entries securely
            X_train, X_test, y_train, y_test = train_test_split(
                X_array, y_array, 
                test_size=test_split_ratio, 
                random_state=42, 
                stratify=y_array if y_array.nunique() > 1 else None
            )
            
            # Process transformations
            X_train_processed = preprocessor_pipeline.fit_transform(X_train)
            X_test_processed = preprocessor_pipeline.transform(X_test)
            
            # Configure algorithms with robust hyperparameters
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
                # Fit model
                classifier_instance.fit(X_train_processed, y_train)
                
                # Predict
                train_predictions = classifier_instance.predict(X_train_processed)
                test_predictions = classifier_instance.predict(X_test_processed)
                
                if hasattr(classifier_instance, "predict_proba"):
                    test_probabilities = classifier_instance.predict_proba(X_test_processed)[:, 1]
                else:
                    test_probabilities = classifier_instance.decision_function(X_test_processed)
                
                # Metrics Calculation
                score_train_acc = accuracy_score(y_train, train_predictions)
                score_test_acc = accuracy_score(y_test, test_predictions)
                score_precision = precision_score(y_test, test_predictions, zero_division=0)
                score_recall = recall_score(y_test, test_predictions, zero_division=0)
                score_f1 = f1_score(y_test, test_predictions, zero_division=0)
                
                performance_metrics_store.append({
                    'Model Configuration': model_name,
                    'Train Accuracy': score_train_acc,
                    'Test Accuracy': score_test_acc,
                    'Precision': score_precision,
                    'Recall': score_recall,
                    'F1-Score': score_f1
                })
                
                # ROC Performance Metrics Mapping
                false_positive_rate, true_positive_rate, _ = roc_curve(y_test, test_probabilities)
                roc_curve_plotting_store[model_name] = (false_positive_rate, true_positive_rate, auc(false_positive_rate, true_positive_rate))
                
                # Confusion Matrix Mapping
                confusion_matrix_plotting_store[model_name] = confusion_matrix(y_test, test_predictions)
                
            metrics_dataframe = pd.DataFrame(performance_metrics_store)
            
            # Save metrics to Session State to maintain continuity
            st.session_state.dynamic_ml_memory = {
                'metrics_table': metrics_dataframe,
                'roc_data': roc_curve_plotting_store,
                'matrix_data': confusion_matrix_plotting_store,
                'class_labels': target_label_encoder.classes_.tolist()
            }
            
    # Render ML results if cached in Memory
    if st.session_state.dynamic_ml_memory is not None:
        saved_ml = st.session_state.dynamic_ml_memory
        
        st.subheader("📊 Comparative Algorithm Performance Matrix")
        st.dataframe(saved_ml['metrics_table'].style.format({
            'Train Accuracy': "{:.2%}", 'Test Accuracy': "{:.2%}",
            'Precision': "{:.2%}", 'Recall': "{:.2%}", 'F1-Score': "{:.2%}"
        }).background_gradient(cmap='Blues'))
        
        # Melt dataframe to generate dynamic visualizations
        melted_metrics = saved_ml['metrics_table'].melt(id_vars='Model Configuration', var_name='Performance Metric', value_name='Score Summary')
        fig_comparative_metrics = px.bar(
            melted_metrics, 
            x='Model Configuration', 
            y='Score Summary', 
            color='Performance Metric', 
            barmode='group',
            title="Parallel Algorithm Performance Comparison Benchmarks",
            color_discrete_sequence=px.colors.qualitative.Bold,
            template="plotly_white"
        )
        st.plotly_chart(fig_comparative_metrics, use_container_width=True)
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown("#### 🔄 Receiver Operating Characteristic (ROC) Space Curves")
            fig_roc_space = go.Figure()
            for name, (fpr, tpr, roc_auc_val) in saved_ml['roc_data'].items():
                fig_roc_space.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC = {roc_auc_val:.3f})"))
            fig_roc_space.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='darkgrey'), name='Random Anchor Base'))
            fig_roc_space.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", template="plotly_white", height=500)
            st.plotly_chart(fig_roc_space, use_container_width=True)
            
        with col_v2:
            st.markdown("#### 🧩 Confusion Matrices Evaluation Matrix")
            fig_plt_cm, axes_grid = plt.subplots(2, 2, figsize=(10, 8))
            axes_grid = axes_grid.ravel()
            
            for idx_pointer, (name, cm_matrix) in enumerate(saved_ml['matrix_data'].items()):
                sns.heatmap(cm_matrix, annot=True, fmt='d', ax=axes_grid[idx_pointer], cmap='Blues',
                            xticklabels=saved_ml['class_labels'], yticklabels=saved_ml['class_labels'], cbar=False)
                axes_grid[idx_pointer].set_title(name, fontsize=10, fontweight='bold')
                axes_grid[idx_pointer].set_xlabel('Predicted Label', fontsize=8)
                axes_grid[idx_pointer].set_ylabel('True Label', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_plt_cm)
    else:
        st.info("💡 Click the button above to execute feature engineering pipelines and train the classifiers.")

# =======================================================================================
# TAB 4: EXECUTIVE FINDINGS & ACTIONS (100% COMPLETELY DYNAMIC TEXT ENGINE)
# =======================================================================================
with tab4:
    st.header("📝 Automated Strategic Briefing & Action Register")
    st.write("This briefing is generated dynamically from the active data matrix and model runs to provide leadership with actionable insights.")
    
    # Run structural descriptive statistics to back written briefs dynamically
    if 'AGE_BINS' in df.columns:
        dynamic_age_crosstab = pd.crosstab(df['AGE_BINS'], df['POLICY_STATUS'], normalize='index') * 100
        all_rep_cols = [c for c in dynamic_age_crosstab.columns if 'REPUD' in str(c) or 'DENI' in str(c) or 'REJ' in str(c)]
        target_rep_col = all_rep_cols[0] if all_rep_cols else dynamic_age_crosstab.columns[-1]
        highest_age_anomaly = dynamic_age_crosstab.sort_values(by=target_rep_col, ascending=False).index[0]
        highest_age_anomaly_rate = dynamic_age_crosstab.sort_values(by=target_rep_col, ascending=False)[target_rep_col].iloc[0]
    else:
        highest_age_anomaly, highest_age_anomaly_rate = "UNKNOWN", 0
        
    if 'ZONE' in df.columns:
        dynamic_zone_crosstab = pd.crosstab(df['ZONE'], df['POLICY_STATUS'], normalize='index') * 100
        all_rep_cols = [c for c in dynamic_zone_crosstab.columns if 'REPUD' in str(c) or 'DENI' in str(c) or 'REJ' in str(c)]
        target_rep_col = all_rep_cols[0] if all_rep_cols else dynamic_zone_crosstab.columns[-1]
        highest_zone_anomaly = dynamic_zone_crosstab.sort_values(by=target_rep_col, ascending=False).index[0]
        highest_zone_anomaly_rate = dynamic_zone_crosstab.sort_values(by=target_rep_col, ascending=False)[target_rep_col].iloc[0]
    else:
        highest_zone_anomaly, highest_zone_anomaly_rate = "UNKNOWN", 0

    mean_income_approved_calc = df[df['POLICY_STATUS'] == primary_approved_str]['PI_ANNUAL_INCOME'].mean() if 'PI_ANNUAL_INCOME' in df.columns else 0
    mean_income_repudiated_calc = df[df['POLICY_STATUS'] != primary_approved_str]['PI_ANNUAL_INCOME'].mean() if 'PI_ANNUAL_INCOME' in df.columns and len(df[df['POLICY_STATUS'] != primary_approved_str]) > 0 else 0

    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
    st.subheader("📌 Active Operational Insights Summary")
    
    st.markdown(f"""
    - **Regional Process Disparities:** The data reveals that claims processing results vary significantly by region. 
      The group operating under **{highest_zone_anomaly}** exhibits the highest rejection footprint, showing a **{highest_zone_anomaly_rate:.2f}% repudiation rate**. 
      This wide regional gap indicates that localized underwriting teams may be applying different evaluation standards.
    - **Socio-Economic Income Gaps:** Approved claims carry a mean annual income profile of **₹{mean_income_approved_calc:,.2f}**, 
      while rejected accounts show an average profile of **₹{mean_income_repudiated_calc:,.2f}**. 
      This trend indicates that lower-income policyholders face higher friction during the claim settlement process.
    - **Age-Based Risk Variances:** Systemic monitoring algorithms have flagged maximum friction within the **{highest_age_anomaly}** cohort, 
      where the repudiation rate reaches **{highest_age_anomaly_rate:.2f}%**. This variance highlights the need to ensure that young or elderly policyholders are evaluated under equal criteria.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("🤖 Algorithmic System Audit")
    if st.session_state.dynamic_ml_memory is not None:
        saved_metrics_brief = st.session_state.dynamic_ml_memory['metrics_table']
        champion_model_row = saved_metrics_brief.sort_values(by='Test Accuracy', ascending=False).iloc[0]
        
        st.info(f"🚀 **Model Audit Finding:** Based on the latest evaluation run, the **{champion_model_row['Model Configuration']}** algorithm is the top-performing model, achieving a validation accuracy of **{champion_model_row['Test Accuracy']:.2%}** and an F1-Score of **{champion_model_row['F1-Score']:.2%}**. Its high predictive capacity suggests that claim outcomes follow highly consistent data patterns rather than purely random variations.")
    else:
        st.warning("⚠️ Run the predictive workflows under the 'Super-Learning Classifiers' tab to populate this section with real-time model metrics.")

    st.subheader("🛠️ Strategic Action Plan for Leadership")
    st.markdown("""
    1. **Standardize Adjudication Rules:** Implement centralized verification workflows to eliminate the localized process differences observed in high-rejection regions.
    2. **Refine Non-Medical Underwriting Filters:** Review and update screening criteria for non-medical policies to prevent verification friction from causing downstream rejections at the claim stage.
    3. **Deploy Automated Quality Audits:** Use the top-performing super-learning model as a secondary quality check to automatically flag high-risk claim rejections for independent manual review.
    """)
