import streamlit as st
import pandas as pd
import numpy as np
import shap
import time
import os
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from interpretable_ids_chat import InterpretableIDSChat

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


@st.cache_resource
def load_interpretable_assistant():
    kb_dir = os.path.join("knowledge_base", "attacks")
    return InterpretableIDSChat(
        kb_dir=kb_dir,
        model="llama3.2:latest",
        ollama_base_url="http://localhost:11434",
    )

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

def transform_features(input_df):

    X = input_df[selected_features]

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

    return X_scaled


def preprocessing(sample):

    start = time.time()

    X_scaled = transform_features(sample)

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

    label = 1 if attack_prob > 0.85 else 0

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

@st.cache_data
def load_shap_background_data(max_rows=64):
    # Prefer the larger UNSW file for more representative background;
    # fall back to demo samples when unavailable.
    candidate_paths = [
        "data/UNSW_NB15_testing-set.csv",
        "data/unsw_test_samples.csv"
    ]

    bg_df = None
    for path in candidate_paths:
        try:
            temp_df = pd.read_csv(path)
            bg_df = temp_df
            break
        except Exception:
            continue

    if bg_df is None:
        raise RuntimeError("Unable to load background data for SHAP explainer.")

    # Ensure expected feature columns exist in-order.
    bg_df = bg_df.reindex(columns=selected_features)
    bg_df = bg_df.dropna(how="all")

    if len(bg_df) == 0:
        raise RuntimeError("Background data has no usable rows for SHAP explainer.")

    if len(bg_df) > max_rows:
        bg_df = bg_df.sample(n=max_rows, random_state=42)

    return bg_df.reset_index(drop=True)


@st.cache_resource
def load_cached_shap_explainer():
    bg_df = load_shap_background_data(max_rows=64)
    bg_scaled = transform_features(bg_df)
    explainer = shap.DeepExplainer(dnn, bg_scaled)
    return explainer


try:
    cached_shap_explainer = load_cached_shap_explainer()
    cached_shap_explainer_error = None
except Exception as e:
    cached_shap_explainer = None
    cached_shap_explainer_error = str(e)

def shap_explain(X):
    if cached_shap_explainer is None:
        raise RuntimeError(
            f"SHAP explainer is not initialized. {cached_shap_explainer_error}"
        )

    X_np = np.asarray(X, dtype=float)
    shap_values = cached_shap_explainer.shap_values(X_np)

    # Normalize output shape for binary TF model.
    if isinstance(shap_values, list):
        shap_arr = np.asarray(shap_values[0])
    else:
        shap_arr = np.asarray(shap_values)

    if shap_arr.ndim == 3:
        shap_arr = shap_arr[:, :, 0]

    # Plot top signed contributions for the current sample.
    sample_vals = shap_arr[0]
    abs_vals = np.abs(sample_vals)
    top_k = min(15, len(sample_vals))
    top_idx = np.argsort(abs_vals)[-top_k:]
    plot_vals = sample_vals[top_idx]
    plot_feats = np.array(selected_features)[top_idx]
    order = np.argsort(np.abs(plot_vals))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in plot_vals[order]]
    ax.barh(plot_feats[order], plot_vals[order], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("SHAP - Top feature contributions (DeepExplainer)")
    ax.set_xlabel("SHAP value (signed)")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()

    top_features = []
    for feat, val in zip(plot_feats[order], plot_vals[order]):
        top_features.append({"feature": str(feat), "shap_value": float(val)})

    return fig, top_features

# ------------------------------
# RUN PIPELINE
# ------------------------------

TOTAL_STEPS = 7
STEP_TITLES = {
    1: "Data Preprocessing",
    2: "Intrusion Detection using DNN",
    3: "Extract Embedding",
    4: "Identify Cluster",
    5: "Identify Attack Category",
    6: "Model Explainability (SHAP)",
    7: "Runtime Analysis"
}


def reset_pipeline_state():
    st.session_state.current_step = 0
    st.session_state.pipeline_values = {}
    st.session_state.step_times = {}
    st.session_state.ids_summary = None
    st.session_state.ids_sources = []
    st.session_state.ids_chat_messages = []
    st.session_state.ids_context_sig = None
    st.session_state.show_ai_page = False


def build_llm_pipeline_context(sample_row, vals, times):
    sample_features = sample_row[selected_features].iloc[0].to_dict()
    attack_probability = float(vals.get("attack_prob", 0.0))

    shap_with_values = []
    for item in vals.get("shap_top_features", []):
        feat = item.get("feature")
        shap_val = float(item.get("shap_value", 0.0))
        raw_val = sample_features.get(feat)
        shap_with_values.append(
            {
                "feature": feat,
                "shap_value": shap_val,
                "feature_value": raw_val,
            }
        )

    context = {
        "sample_index": int(sample_index),
        "binary_label": "attack" if int(vals.get("label", 0)) == 1 else "normal",
        "attack_probability": attack_probability,
        "cluster": vals.get("cluster", None),
        "attack_name": vals.get("attack_name", "Normal" if int(vals.get("label", 0)) == 0 else "Unknown"),
        "attack_code": int(np.asarray(vals["attack"]).item()) if "attack" in vals else None,
        "step_times": {k: float(v) for k, v in times.items()},
        "shap_top_features": vals.get("shap_top_features", []),
        "shap_feature_evidence": shap_with_values,
        "sample_feature_values": sample_features,
        "interpretation_hints": [
            "Use SHAP sign and magnitude to explain why each top feature pushes toward attack or normal.",
            "Relate notable raw values to potential risk context (for example ttl/window/load style indicators).",
            "Do not rename attack family; keep consistent with pipeline attack_name.",
        ],
    }
    return context


def render_interpretable_assistant_panel(vals, times):
    st.markdown("### Interpretable IDS Assistant (RAG + Ollama)")
    st.caption("Interactive analyst chat grounded in IDS pipeline evidence + RAG knowledge.")

    clear_chat_clicked = st.button("Clear AI Chat")

    if clear_chat_clicked:
        st.session_state.ids_summary = None
        st.session_state.ids_sources = []
        st.session_state.ids_chat_messages = []
        st.rerun()

    llm_context = build_llm_pipeline_context(sample, vals, times)
    context_sig = (
        llm_context.get("sample_index"),
        llm_context.get("binary_label"),
        llm_context.get("attack_name"),
        llm_context.get("attack_code"),
    )

    # Regenerate automatically if context changed or first open.
    if st.session_state.get("ids_context_sig") != context_sig:
        st.session_state.ids_chat_messages = []
        st.session_state.ids_sources = []
        st.session_state.ids_summary = None
        st.session_state.ids_context_sig = context_sig

    if not st.session_state.get("ids_chat_messages"):
        try:
            assistant = load_interpretable_assistant()
            summary, retrieved = assistant.generate_initial_summary(llm_context)

            st.session_state.ids_sources = [
                {
                    "source": r.source,
                    "score": float(r.score),
                    "snippet": r.text[:240],
                }
                for r in retrieved
            ]
            st.session_state.ids_chat_messages = [
                {"role": "assistant", "content": summary}
            ]
        except Exception as e:
            st.error(f"AI summary generation failed: {e}")

    with st.expander("RAG Sources Used"):
        for src in st.session_state.get("ids_sources", []):
            st.markdown(
                f"- `{src['source']}` (score: {src['score']:.4f})\n\n"
                f"  {src['snippet']}..."
            )

    st.markdown("#### Chat")
    for msg in st.session_state.get("ids_chat_messages", []):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_msg = st.chat_input("Ask follow-up about this alert, evidence, or mitigation")
    if user_msg:
        st.session_state.ids_chat_messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.write(user_msg)

        try:
            assistant = load_interpretable_assistant()
            answer, retrieved = assistant.chat_follow_up(
                pipeline_context=llm_context,
                chat_history=st.session_state.ids_chat_messages,
                user_message=user_msg,
            )
            st.session_state.ids_chat_messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)

            st.session_state.ids_sources = [
                {
                    "source": r.source,
                    "score": float(r.score),
                    "snippet": r.text[:240],
                }
                for r in retrieved
            ]
        except Exception as e:
            st.error(f"AI follow-up failed: {e}")


if "current_step" not in st.session_state:
    reset_pipeline_state()

if "sample_key" not in st.session_state:
    st.session_state.sample_key = int(sample_index)

if st.session_state.sample_key != int(sample_index):
    st.session_state.sample_key = int(sample_index)
    reset_pipeline_state()


def render_up_to_step(step_limit):
    if step_limit <= 0:
        return

    vals = st.session_state.pipeline_values
    times = st.session_state.step_times

    if step_limit >= 1:
        st.subheader("Step 1 - Data Preprocessing")
        st.write("Scaling selected features and encoding categorical inputs.")

        if "X_scaled" not in vals:
            X_scaled, t = preprocessing(sample)
            vals["X_scaled"] = X_scaled
            times["Preprocessing"] = t

        st.success(f"Completed in {times['Preprocessing']:.4f} seconds")
        scaled_df = pd.DataFrame(vals["X_scaled"], columns=selected_features)
        st.caption("Scaled feature preview")
        st.dataframe(scaled_df.T.rename(columns={0: "scaled_value"}), use_container_width=True)

    if step_limit >= 2:
        st.subheader("Step 2 - Intrusion Detection using DNN")

        if "label" not in vals:
            label, attack_prob, t = dnn_predict(vals["X_scaled"])
            vals["label"] = label
            vals["attack_prob"] = attack_prob
            times["DNN"] = t

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Attack Probability", f"{vals['attack_prob']:.4f}")
        with c2:
            st.metric("DNN Runtime (s)", f"{times['DNN']:.4f}")

        st.progress(int(max(0.0, min(1.0, vals["attack_prob"])) * 100), text="DNN attack score")

        if vals["label"] == 0:
            st.success("Traffic classified as **NORMAL**")
        else:
            st.error("Attack detected")

    if step_limit >= 3:
        st.subheader("Step 3 - Extract Embedding")

        if vals.get("label", 0) == 0:
            st.info("Traffic is normal, so embedding extraction for attack path is skipped.")
            chart_data = pd.DataFrame(
                sample[selected_features].values[0],
                index=selected_features,
                columns=["Value"]
            )
            st.bar_chart(chart_data)
        else:
            if "emb" not in vals:
                emb, t = extract_embeddings(vals["X_scaled"])
                vals["emb"] = emb
                times["Embedding"] = t

            st.metric("Embedding Runtime (s)", f"{times['Embedding']:.4f}")
            emb_df = pd.DataFrame(vals["emb"][0][:20], columns=["embedding"])
            st.caption("First 20 embedding dimensions")
            st.line_chart(emb_df)

    if step_limit >= 4:
        st.subheader("Step 4 - Identify Cluster")

        if vals.get("label", 0) == 0:
            st.info("Traffic is normal, so cluster assignment is skipped.")
        else:
            if "cluster" not in vals:
                cluster, t = cluster_predict(vals["emb"])
                vals["cluster"] = cluster
                times["Clustering"] = t

            c1, c2 = st.columns(2)
            c1.metric("Cluster", str(vals["cluster"]))
            c2.metric("Clustering Runtime (s)", f"{times['Clustering']:.4f}")

    if step_limit >= 5:
        st.subheader("Step 5 - Identify Attack Category")

        if vals.get("label", 0) == 0:
            st.success("Final category: **Normal** (XGBoost attack classifier skipped).")
        else:
            if "attack" not in vals:
                attack, t = attack_classification(vals["cluster"], vals["X_scaled"], vals["emb"])
                vals["attack"] = attack
                times["XGBoost"] = t

                try:
                    if attack_label_encoder is not None:
                        vals["attack_name"] = attack_label_encoder.inverse_transform([int(np.asarray(attack).item())])[0]
                    else:
                        vals["attack_name"] = f"code: {int(np.asarray(attack).item())}"
                except Exception:
                    vals["attack_name"] = str(attack)

            c1, c2 = st.columns(2)
            c1.metric("XGBoost Runtime (s)", f"{times['XGBoost']:.4f}")
            c2.metric("Cluster Used", str(vals.get("cluster", "N/A")))
            st.error(f"Predicted Attack Type: **{vals['attack_name']}**")

    if step_limit >= 6:
        st.subheader("Step 6 - Model Explainability (SHAP)")

        if "shap_fig" not in vals and "shap_error" not in vals:
            shap_start = time.time()
            try:
                shap_fig, shap_top_features = shap_explain(vals["X_scaled"])
                vals["shap_fig"] = shap_fig
                vals["shap_top_features"] = shap_top_features
            except Exception as e:
                vals["shap_error"] = str(e)
            times["SHAP"] = time.time() - shap_start

        if "shap_fig" in vals:
            st.pyplot(vals["shap_fig"])
        else:
            st.error(f"SHAP explanation failed: {vals.get('shap_error', 'unknown error')}")

        if "SHAP" in times:
            st.caption(f"SHAP Runtime: {times['SHAP']:.4f} seconds")

    if step_limit >= 7:
        st.subheader("Step 7 - Runtime Analysis")

        runtime_df = pd.DataFrame({
            "Step": list(times.keys()),
            "Time (seconds)": list(times.values())
        })
        st.dataframe(runtime_df, use_container_width=True)

        total_runtime = float(np.sum(list(times.values()))) if len(times) > 0 else 0.0
        st.metric("Total Pipeline Runtime", f"{total_runtime:.4f} seconds")
        st.success("Pipeline execution completed")

        c_open, _ = st.columns([1, 3])
        if c_open.button("Open AI Assistant Page"):
            st.session_state.show_ai_page = True
            st.rerun()


def render_assistant_page():
    st.title("Interpretable IDS Assistant")
    st.caption("Dedicated analyst view with back navigation.")

    c_back, _ = st.columns([1, 6])
    if c_back.button("← Back to Pipeline"):
        st.session_state.show_ai_page = False
        st.rerun()

    vals = st.session_state.get("pipeline_values", {})
    times = st.session_state.get("step_times", {})

    if st.session_state.get("current_step", 0) < TOTAL_STEPS:
        st.warning("Complete all pipeline steps first, then use AI assistant.")
        return

    render_interpretable_assistant_panel(vals, times)


if st.session_state.get("show_ai_page", False):
    render_assistant_page()
    st.stop()


if mode == "Step-by-Step":
    st.sidebar.markdown("### Step Controls")
    next_step_clicked = st.sidebar.button("⏭️ Next", help="Run next step")
    go_end_clicked = st.sidebar.button("⏩ Go End", help="Run all remaining steps")
    reset_clicked = st.sidebar.button("🔄 Reset", help="Reset current sample run")

    current_label = STEP_TITLES.get(st.session_state.current_step, "Not started")
    st.sidebar.caption(f"Current step: {st.session_state.current_step}/{TOTAL_STEPS}")
    st.sidebar.caption(current_label)

    if reset_clicked:
        reset_pipeline_state()
        st.rerun()

    if go_end_clicked:
        st.session_state.current_step = TOTAL_STEPS
    elif next_step_clicked and st.session_state.current_step < TOTAL_STEPS:
        st.session_state.current_step += 1

    progress.progress(int((st.session_state.current_step / TOTAL_STEPS) * 100))
    render_up_to_step(st.session_state.current_step)

else:
    st.sidebar.markdown("### Run Controls")
    run_all_clicked = st.sidebar.button("▶️ Run IDS Pipeline")
    reset_all_clicked = st.sidebar.button("🔄 Reset")

    if reset_all_clicked:
        reset_pipeline_state()
        st.rerun()

    if run_all_clicked:
        st.session_state.current_step = TOTAL_STEPS

    progress.progress(int((st.session_state.current_step / TOTAL_STEPS) * 100))
    render_up_to_step(st.session_state.current_step)