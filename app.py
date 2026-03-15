import streamlit as st
import pandas as pd
import numpy as np
import shap
import time
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ------------------------------
# PAGE CONFIG
# ------------------------------

st.set_page_config(
    page_title="Hybrid IDS Demo",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Hybrid Intrusion Detection System")
st.markdown("Deep Neural Network + KMeans Clustering + XGBoost + SHAP Explainability")

# ------------------------------
# LOAD MODELS
# ------------------------------

@st.cache_resource
def load_models():

    dnn = tf.keras.models.load_model("models/global_dnn_model.h5")
    encoder = tf.keras.models.load_model("models/global_encoder_model.h5")

    kmeans = joblib.load("models/kmeans_attack_model.pkl")

    scaler = joblib.load("models/scaler.pkl")

    # Try to load saved label encoders (optional). If not found, we'll build encoders at runtime.
    try:
        label_encoders = joblib.load("models/label_encoders.pkl")
    except Exception:
        label_encoders = None

    # Try to load saved categorical column list (optional)
    try:
        categorical_columns = joblib.load("models/categorical_columns.pkl")
    except Exception:
        categorical_columns = None

    xgb_models = {
        0: joblib.load("models/xgb_attack_cluster_0_multiclass.pkl"),
        1: joblib.load("models/xgb_attack_cluster_1_multiclass.pkl"),
        2: joblib.load("models/xgb_attack_cluster_2_multiclass.pkl")
    }

    # Try to load attack label encoder (maps encoded class -> attack name)
    try:
        attack_label_encoder = joblib.load("models/attack_label_encoder.pkl")
    except Exception:
        attack_label_encoder = None

    return dnn, encoder, kmeans, scaler, xgb_models, label_encoders, categorical_columns, attack_label_encoder


dnn, encoder, kmeans, scaler, xgb_models, label_encoders, categorical_columns, attack_label_encoder = load_models()

# ------------------------------
# LOAD DATA
# ------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/unsw_test_samples.csv")
    return df


df = load_data()

# ------------------------------
# SAMPLE SELECTION
# ------------------------------

st.sidebar.header("Select Test Sample")

sample_index = st.sidebar.selectbox("Sample Index", df.index)

sample = df.loc[[sample_index]]

st.subheader("Selected Sample")
st.dataframe(sample)

# ------------------------------
# FEATURES
# ------------------------------

selected_features = [
'ackdat','ct_dst_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','ct_src_dport_ltm',
'ct_src_ltm','ct_srv_dst','ct_srv_src','ct_state_ttl','dbytes','dinpkt',
'djit','dload','dloss','dmean','dpkts','dtcpb','dttl','dur','proto',
'rate','sbytes','service','sinpkt','sjit','sload','sloss','smean',
'spkts','state','stcpb','sttl','swin','synack','tcprtt'
]

# ------------------------------
# MODE
# ------------------------------

mode = st.radio(
    "Execution Mode",
    ["Step-by-Step", "Run Complete Pipeline"]
)

progress = st.progress(0)

# ------------------------------
# STEP 1 PREPROCESSING
# ------------------------------

def preprocessing(sample):

    start = time.time()

    X = sample[selected_features]

    # --- Encode categorical columns before scaling ---
    X = X.copy()

    # Determine categorical columns to encode:
    if categorical_columns is not None:
        categorical_cols = list(categorical_columns)
    else:
        # fall back to detecting object/category columns in the sample
        categorical_cols =  ['proto','service','state']

    if len(categorical_cols) > 0:

        # If we have saved encoders from training, use them. Otherwise fit on the loaded dataset `df`.
        encoders = {}

        if label_encoders is not None:
            encoders = label_encoders

        for col in categorical_cols:

            if encoders and col in encoders:
                le = encoders[col]
                # transform; handle unseen labels by mapping to -1 then to a numeric value
                try:
                        # ensure strings when using LabelEncoder
                        X[col] = le.transform(X[col].astype(str).values)
                except Exception:
                    # fallback: fit encoder on global df's column then transform
                    le = LabelEncoder()
                    le.fit(df[col].astype(str).values)
                    X[col] = le.transform(X[col].astype(str).values)
                    encoders[col] = le
            else:
                # build encoder from the available dataset `df` (best-effort)
                le = LabelEncoder()
                le.fit(df[col].astype(str).values)
                X[col] = le.transform(X[col].astype(str).values)
                encoders[col] = le

        # If we built encoders at runtime, keep them in memory (not persisted here).

    # Ensure all columns are numeric (coerce any remaining strings)
    X = X.apply(lambda c: pd.to_numeric(c, errors='coerce'))

    # Fill any NaNs produced by coercion with 0 to avoid scaler errors (best-effort)
    X = X.fillna(0)

    X_scaled = scaler.transform(X)

    runtime = time.time() - start

    return X_scaled, runtime

# ------------------------------
# STEP 2 DNN PREDICTION
# ------------------------------
# ------------------------------
# STEP 2 DNN PREDICTION
# ------------------------------

def dnn_predict(X):

    start = time.time()

    pred = dnn.predict(X, verbose=0)

    # binary classifier output
    attack_prob = float(pred[0][0])

    label = 1 if attack_prob > 0.8 else 0

    runtime = time.time() - start

    return label, attack_prob, runtime

# ------------------------------
# STEP 3 EMBEDDINGS
# ------------------------------

def extract_embeddings(X):

    start = time.time()

    emb = encoder.predict(X, verbose=0)

    runtime = time.time() - start

    return emb, runtime

# ------------------------------
# STEP 4 CLUSTERING
# ------------------------------

def cluster_predict(emb):

    start = time.time()

    cluster = kmeans.predict(emb)[0]

    runtime = time.time() - start

    return cluster, runtime

# ------------------------------
# STEP 5 ATTACK CLASSIFICATION
# ------------------------------

def attack_classification(cluster, X, emb=None):

    start = time.time()

    model = xgb_models[cluster]

    # X should be the same feature set the XGBoost models were trained on.
    # During training we concatenated scaled features + embeddings. Recreate that here.
    try:
        if emb is not None:
            # ensure numpy arrays and proper shapes
            X_in = np.hstack((np.asarray(X), np.asarray(emb)))
        else:
            X_in = np.asarray(X)

        attack = model.predict(X_in)[0]
    except Exception as e:
        # provide a clearer error for troubleshooting
        raise ValueError(f"XGBoost prediction failed: {e}")

    runtime = time.time() - start

    return attack, runtime

# ------------------------------
# STEP 6 SHAP
# ------------------------------

def shap_explain(X):

    # Convert input to numpy
    X_np = np.array(X)

    # Wrapper so SHAP receives 1D output
    def predict_fn(x):
        return dnn.predict(x, verbose=0).flatten()

    # Small background dataset
    background = np.zeros((1, X_np.shape[1]))

    explainer = shap.KernelExplainer(predict_fn, background)

    shap_values = explainer.shap_values(X_np)

    # Ensure numpy array
    shap_values = np.array(shap_values)

    # Build explicit matplotlib figure
    fig, ax = plt.subplots(figsize=(10,4))

    shap.bar_plot(
        shap_values[0],
        feature_names=selected_features,
        max_display=15
    )

    plt.tight_layout()

    return fig

# ------------------------------
# RUN PIPELINE
# ------------------------------

if st.button("Run IDS Pipeline"):

    total_start = time.time()

    step_times = {}

    # ------------------
    # STEP 1
    # ------------------

    progress.progress(10)

    st.subheader("Step 1 — Data Preprocessing")

    st.write("Scaling selected features using trained StandardScaler.")

    X_scaled, t = preprocessing(sample)

    step_times["Preprocessing"] = t

    st.success(f"Completed in {t:.4f} seconds")

    # ------------------
    # STEP 2
    # ------------------

    progress.progress(25)

    st.subheader("Step 2 — Intrusion Detection using DNN")

    label, attack_prob, t = dnn_predict(X_scaled)

    st.write(f"Attack probability: {attack_prob:.4f}")

    step_times["DNN"] = t

    if label == 0:

        st.success("Traffic classified as **NORMAL**")

    else:

        st.error("⚠️ Attack Detected")

    st.write(f"Runtime: {t:.4f} seconds")

    # ------------------
    # NORMAL TRAFFIC
    # ------------------

    if label == 0:

        progress.progress(60)

        st.subheader("Normal Traffic Visualization")

        chart_data = pd.DataFrame(
            sample[selected_features].values[0],
            index=selected_features,
            columns=["Value"]
        )

        st.bar_chart(chart_data)

    # ------------------
    # ATTACK PIPELINE
    # ------------------

    else:

        progress.progress(45)

        st.subheader("Step 3 — Extracting Feature Embeddings")

        emb, t = extract_embeddings(X_scaled)

        step_times["Embedding"] = t

        st.write(f"Runtime: {t:.4f} seconds")

        progress.progress(60)

        st.subheader("Step 4 — KMeans Cluster Identification")

        cluster, t = cluster_predict(emb)

        step_times["Clustering"] = t

        st.success(f"Cluster Assigned: **{cluster}**")

        st.write(f"Runtime: {t:.4f} seconds")

        progress.progress(75)

        st.subheader("Step 5 — Cluster-Specific Attack Classification")

        # Pass embeddings along with scaled features so XGBoost receives the same feature layout used in training
        attack, t = attack_classification(cluster, X_scaled, emb)

        step_times["XGBoost"] = t

        # Map numeric attack code to human-readable label if encoder available
        try:
            if attack_label_encoder is not None:
                try:
                    attack_name = attack_label_encoder.inverse_transform([int(attack)])[0]
                except Exception:
                    # if attack is a numpy array or nested, coerce to int
                    attack_name = attack_label_encoder.inverse_transform([int(np.asarray(attack).item())])[0]
                st.error(f"Predicted Attack Type: **{attack_name}**")
            else:
                st.error(f"Predicted Attack Type (code): **{int(attack)}**")
        except Exception:
            st.error(f"Predicted Attack Type (raw): **{attack}**")

        st.write(f"Runtime: {t:.4f} seconds")

    # ------------------
    # SHAP
    # ------------------

    progress.progress(90)

    st.subheader("Step 6 — Model Explainability (SHAP)")

    try:
        shap_fig = shap_explain(X_scaled)
        st.pyplot(shap_fig)
    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")

    # ------------------
    # RUNTIME
    # ------------------

    progress.progress(100)

    total_runtime = time.time() - total_start

    st.subheader("Runtime Analysis")

    runtime_df = pd.DataFrame({
        "Step": step_times.keys(),
        "Time (seconds)": step_times.values()
    })

    st.dataframe(runtime_df)

    st.metric("Total Pipeline Runtime", f"{total_runtime:.4f} seconds")

    st.success("Pipeline Execution Completed")