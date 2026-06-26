import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Insurance Audit Dashboard", layout="wide", page_icon="🛡️")

st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>🛡️ Claim Settlement Audit Dashboard</h1>
    <hr style='border: 1px solid #AED6F1;'>
""", unsafe_allow_html=True)

# --- DATA LOADING & CLEANING ---
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv('Insurance.csv')
    
    # Clean numeric columns
    df['SUM_ASSURED'] = df['SUM_ASSURED'].replace('[\$,]', '', regex=True).astype(float)
    df['PI_ANNUAL_INCOME'] = df['PI_ANNUAL_INCOME'].astype(str).str.replace(',', '').astype(float)
    
    # Safely fill missing values based on data type
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna('Unknown')
    
    return df

df = load_and_prepare_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.header("Navigation")
menu = ["📊 Descriptive & Diagnostic", "🤖 Machine Learning Audit"]
choice = st.sidebar.radio("Go to:", menu)

# --- MODE 1: DESCRIPTIVE & DIAGNOSTIC ---
if choice == "📊 Descriptive & Diagnostic":
    st.header("Diagnostic Analysis")
    
    # Top-level Metrics
    total_claims = len(df)
    approved_claims = len(df[df['POLICY_STATUS'] == 'Approved Death Claim'])
    repudiated_claims = len(df[df['POLICY_STATUS'] == 'Repudiate Death'])
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Claims Processed", f"{total_claims:,}")
    m2.metric("Approved Claims", f"{approved_claims:,}", f"{(approved_claims/total_claims)*100:.1f}%")
    m3.metric("Repudiated Claims", f"{repudiated_claims:,}", f"-{(repudiated_claims/total_claims)*100:.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Claim Status by Zone")
        # Prepare data for interactive Plotly bar chart
        zone_pivot = pd.crosstab(df['ZONE'], df['POLICY_STATUS'], normalize='index').reset_index()
        fig_bar = px.bar(
            zone_pivot, 
            x='ZONE', 
            y=['Approved Death Claim', 'Repudiate Death'] if 'Repudiate Death' in zone_pivot.columns else zone_pivot.columns[1:],
            title="Proportion of Claims by Zone (1.0 = 100%)",
            barmode='stack',
            color_discrete_sequence=['#2ECC71', '#E74C3C']
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("Income Distribution by Status")
        # Interactive Plotly boxplot
        fig_box = px.box(
            df, 
            x='POLICY_STATUS', 
            y='PI_ANNUAL_INCOME', 
            color='POLICY_STATUS',
            title="Annual Income vs. Policy Status",
            color_discrete_sequence=['#2ECC71', '#E74C3C']
        )
        st.plotly_chart(fig_box, use_container_width=True)

# --- MODE 2: MACHINE LEARNING AUDIT ---
elif choice == "🤖 Machine Learning Audit":
    st.header("Supervised Learning Classification")
    st.markdown("Identify patterns in repudiations using predictive modeling.")
    
    # Preprocessing
    le_target = LabelEncoder()
    df['TARGET'] = le_target.fit_transform(df['POLICY_STATUS'])
    
    # Drop identifiers and encode features
    df_ml = df.drop(['POLICY_NO', 'PI_NAME', 'REASON_FOR_CLAIM', 'POLICY_STATUS'], axis=1)
    le_features = LabelEncoder()
    for col in df_ml.select_dtypes(include=['object']).columns:
        df_ml[col] = le_features.fit_transform(df_ml[col].astype(str))
    
    # Splitting and Scaling
    X = df_ml.drop('TARGET', axis=1)
    y = df_ml['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # UI Controls
    col_model, col_btn = st.columns([3, 1])
    with col_model:
        model_name = st.selectbox("Select Classification Algorithm", ["Random Forest", "Decision Tree", "KNN", "Gradient Boosting"])
    
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    
    with col_btn:
        st.write("") # Spacing
        st.write("")
        run_model = st.button("🚀 Train & Evaluate", use_container_width=True)
    
    if run_model:
        with st.spinner(f'Training {model_name}...'):
            clf = models[model_name]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            st.markdown("---")
            st.subheader(f"Results for {model_name}")
            
            # Layout for metrics
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.write("**Confusion Matrix**")
                cm = confusion_matrix(y_test, y_pred)
                target_names = le_target.classes_
                fig_cm = px.imshow(
                    cm, 
                    text_auto=True, 
                    color_continuous_scale='Blues',
                    x=target_names, 
                    y=target_names,
                    labels=dict(x="Predicted Status", y="Actual Status", color="Count")
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                
            with res_col2:
                st.write("**Classification Report**")
                report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.background_gradient(cmap='Blues'), use_container_width=True)

            # ROC Curve
            if hasattr(clf, "predict_proba"):
                st.write("**ROC Curve & AUC**")
                y_prob = clf.predict_proba(X_test)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                fig_roc = px.area(
                    x=fpr, y=tpr,
                    title=f'ROC Curve (Area under Curve = {roc_auc:.4f})',
                    labels=dict(x='False Positive Rate', y='True Positive Rate'),
                    color_discrete_sequence=['#8E44AD']
                )
                fig_roc.add_shape(
                    type='line', line=dict(dash='dash', color='gray'),
                    x0=0, x1=1, y0=0, y1=1
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.warning("ROC Curve is not available for this model configuration.")