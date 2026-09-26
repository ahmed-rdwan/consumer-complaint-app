import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

st.set_page_config(page_title="Consumer Complaint Classifier", page_icon="📋", layout="centered")

MODEL_REPO = "AhmedRdwan/consumer-complaint-classifier"
LABELS = ["credit_card", "credit_reporting", "debt_collection", "mortgages_and_loans", "retail_banking"]
LABEL_DISPLAY = {
    "credit_card": "Credit Card",
    "credit_reporting": "Credit Reporting",
    "debt_collection": "Debt Collection",
    "mortgages_and_loans": "Mortgages & Loans",
    "retail_banking": "Retail Banking"
}
MAX_LEN = 250


@st.cache_resource   # يحمّل الموديل مرة واحدة بس، مش كل مرة المستخدم يعمل تفاعل
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model.eval()
    return model, tokenizer


model, tokenizer = load_model()
device = torch.device("cpu")


def classify_complaint(text):
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
    inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
    return probs


# --- الواجهة ---
st.title("📋 Consumer Complaint Classifier")
st.markdown(
    "Automatically routes financial consumer complaints to the correct category "
    "using a fine-tuned **DistilBERT** model, trained on CFPB complaint data."
)

examples = [
    "I have been trying to dispute an incorrect entry on my credit report for months and no one responds.",
    "A debt collector keeps calling me multiple times a day about a debt I already paid off.",
    "My mortgage payment increased suddenly without any explanation from the lender.",
]

chosen_example = st.selectbox("Try an example (optional):", ["-- Select an example --"] + examples)
default_text = "" if chosen_example == "-- Select an example --" else chosen_example

text_input = st.text_area("Complaint Narrative", value=default_text, height=180,
                           placeholder="Paste or type a consumer complaint here...")

if st.button("Classify Complaint", type="primary"):
    if not text_input.strip():
        st.warning("Please enter a complaint to classify.")
    else:
        with st.spinner("Analyzing..."):
            probs = classify_complaint(text_input)

        top_index = probs.argmax()
        top_label = LABEL_DISPLAY[LABELS[top_index]]
        top_confidence = probs[top_index]

        st.success(f"**Predicted Category:** {top_label}  \n**Confidence:** {top_confidence:.1%}")

        st.markdown("#### Probability by category")
        sorted_pairs = sorted(zip(LABELS, probs), key=lambda x: x[1], reverse=True)
        for label, prob in sorted_pairs:
            st.write(LABEL_DISPLAY[label])
            st.progress(float(prob))

st.markdown("---")
st.caption(
    "Model: [DistilBERT fine-tuned on CFPB complaints](https://huggingface.co/AhmedRdwan/consumer-complaint-classifier) &nbsp;|&nbsp; "
    "[Source code & training notebook](https://github.com/AhmedRdwan/consumer-complaint-app)"
)