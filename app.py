import streamlit as st
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

st.set_page_config(page_title="Consumer Complaint Classifier", page_icon="📋", layout="wide")

MODEL_REPO = "AhmedRdwan/consumer-complaint-classifier"
MAX_LEN = 250

CATEGORIES = {
    "credit_reporting": {
        "display": "Credit Reporting",
        "icon": "📊",
        "color": "#4C72B0",
        "desc": "Errors on a credit report, disputed entries, or issues with credit bureaus."
    },
    "debt_collection": {
        "display": "Debt Collection",
        "icon": "📞",
        "color": "#DD8452",
        "desc": "Repeated collection calls, disputed debts, or aggressive collector behavior."
    },
    "mortgages_and_loans": {
        "display": "Mortgages & Loans",
        "icon": "🏠",
        "color": "#55A868",
        "desc": "Mortgages, car loans, payday loans, and student loan issues."
    },
    "credit_card": {
        "display": "Credit Card",
        "icon": "💳",
        "color": "#C44E52",
        "desc": "Unauthorized charges, billing disputes, or credit card account issues."
    },
    "retail_banking": {
        "display": "Retail Banking",
        "icon": "🏦",
        "color": "#8172B2",
        "desc": "Checking/savings accounts, money transfers, and services like Venmo."
    },
}
LABELS = list(CATEGORIES.keys())


# ---------- Load model (cached, runs once) ----------
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model.eval()
    return model, tokenizer


model, tokenizer = load_model()


def classify_complaint(text):
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
    inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
    return probs


# ---------- Custom styling ----------
st.markdown("""
<style>
.category-card {
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    height: 160px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    border: 1px solid rgba(128,128,128,0.25);
}
.category-card .icon { font-size: 2em; }
.category-card .title { font-weight: 700; margin-top: 6px; }
.category-card .desc { font-size: 0.8em; color: gray; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)


# ---------- Session state for history ----------
if "history" not in st.session_state:
    st.session_state.history = []


# ---------- Header ----------
st.title("📋 Consumer Complaint Classifier")
st.markdown(
    "A fine-tuned **DistilBERT** model that reads a consumer's complaint and automatically "
    "routes it to the right team — instead of a human manually tagging thousands of complaints. "
    "Trained on real CFPB (Consumer Financial Protection Bureau) complaint data."
)

st.markdown("#### The 5 categories it chooses between")
cols = st.columns(5)
for col, key in zip(cols, LABELS):
    cat = CATEGORIES[key]
    with col:
        st.markdown(
            f"""
            <div class="category-card" style="background-color:{cat['color']}18; border-color:{cat['color']}55;">
                <div class="icon">{cat['icon']}</div>
                <div class="title">{cat['display']}</div>
                <div class="desc">{cat['desc']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("")

# ---------- Tabs ----------
tab_classify, tab_history, tab_about = st.tabs(["🔍 Classify a Complaint", "🕓 Session History", "ℹ️ About the Model"])

# --- Tab 1: Classify ---
with tab_classify:
    examples = [
        "I have been trying to dispute an incorrect entry on my credit report for months and no one responds.",
        "A debt collector keeps calling me multiple times a day about a debt I already paid off.",
        "My mortgage payment increased suddenly without any explanation from the lender.",
        "Someone made an unauthorized charge on my credit card and the bank refuses to refund me.",
        "I can't access my checking account online and customer service is unhelpful.",
    ]

    chosen_example = st.selectbox("Try an example, or write your own below:", ["-- Write my own --"] + examples)
    default_text = "" if chosen_example == "-- Write my own --" else chosen_example

    text_input = st.text_area("Complaint Narrative", value=default_text, height=160,
                               placeholder="Paste or type a consumer complaint here...")

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        classify_clicked = st.button("Classify", type="primary", use_container_width=True)
    with col_info:
        if text_input.strip():
            st.caption(f"{len(text_input.split())} words")

    if classify_clicked:
        if not text_input.strip():
            st.warning("Please enter a complaint to classify.")
        else:
            with st.spinner("Reading the complaint..."):
                probs = classify_complaint(text_input)

            top_index = probs.argmax()
            top_key = LABELS[top_index]
            top_cat = CATEGORIES[top_key]
            top_confidence = probs[top_index]

            # Save to session history
            st.session_state.history.append({
                "Complaint": text_input[:60] + ("..." if len(text_input) > 60 else ""),
                "Predicted": top_cat["display"],
                "Confidence": f"{top_confidence:.1%}"
            })

            st.markdown("---")
            result_col, chart_col = st.columns([1, 1.4])

            with result_col:
                st.markdown(
                    f"""
                    <div class="category-card" style="background-color:{top_cat['color']}25; border-color:{top_cat['color']}; height:auto; padding:24px;">
                        <div class="icon">{top_cat['icon']}</div>
                        <div class="title" style="font-size:1.3em;">{top_cat['display']}</div>
                        <div class="desc" style="font-size:1em; margin-top:10px;">Confidence: <b>{top_confidence:.1%}</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if top_confidence < 0.5:
                    st.caption("⚠️ The model isn't very confident — the complaint may touch multiple categories.")

            with chart_col:
                st.markdown("**Probability across all categories**")
                chart_df = pd.DataFrame({
                    "Category": [CATEGORIES[l]["display"] for l in LABELS],
                    "Probability": probs
                }).sort_values("Probability", ascending=True)
                st.bar_chart(chart_df.set_index("Category"), horizontal=True)

# --- Tab 2: History ---
with tab_history:
    if not st.session_state.history:
        st.info("No complaints classified yet this session — try one in the 'Classify a Complaint' tab.")
    else:
        st.markdown(f"**{len(st.session_state.history)} complaint(s) classified this session:**")
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

# --- Tab 3: About ---
with tab_about:
    st.markdown("""
    ### How this model was built

    Four architectures were trained from scratch or fine-tuned on ~100k+ real consumer
    complaints and compared using Macro F1 (fairer than raw accuracy on this imbalanced dataset):

    | Model | Macro F1 |
    |---|---|
    | SimpleRNN | 0.16 |
    | LSTM | 0.82 |
    | GRU | 0.83 |
    | **DistilBERT (final)** | **0.84** |

    DistilBERT — a pretrained transformer, fine-tuned on this dataset — was selected as the
    final model for its strongest performance and better generalization to real, freeform text.

    **Links:**
    - 🤗 [Model on Hugging Face](https://huggingface.co/AhmedRdwan/consumer-complaint-classifier)
    - 💻 [Source code & training notebook](https://github.com/AhmedRdwan/consumer-complaint-app)
    """)

st.markdown("---")
st.caption("Built as a portfolio NLP project — not affiliated with the CFPB.")