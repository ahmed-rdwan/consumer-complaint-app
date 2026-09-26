import os
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

MODEL_REPO = "AhmedRdwan/consumer-complaint-classifier"

app_model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
app_tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app_model.to(device)
app_model.eval()

LABELS = ["credit_card", "credit_reporting", "debt_collection", "mortgages_and_loans", "retail_banking"]
LABEL_DISPLAY = {
    "credit_card": "Credit Card",
    "credit_reporting": "Credit Reporting",
    "debt_collection": "Debt Collection",
    "mortgages_and_loans": "Mortgages & Loans",
    "retail_banking": "Retail Banking"
}
MAX_LEN = 250


def classify_complaint(text):
    if not text or not text.strip():
        return {LABEL_DISPLAY[l]: 0.0 for l in LABELS}, "Please enter a complaint to classify."

    inputs = app_tokenizer(
        text, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}

    with torch.no_grad():
        outputs = app_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    result = {LABEL_DISPLAY[label]: float(prob) for label, prob in zip(LABELS, probs)}
    top_label = LABEL_DISPLAY[LABELS[probs.argmax()]]
    top_confidence = float(probs.max())

    status = f"**Predicted Category:** {top_label}  \n**Confidence:** {top_confidence:.1%}"
    return result, status


custom_css = """
.gradio-container { font-family: 'Inter', -apple-system, sans-serif; max-width: 900px !important; margin: auto; }
#header { text-align: center; padding: 10px 0 20px 0; }
#header h1 { font-size: 1.8em; margin-bottom: 4px; }
#header p { color: #666; font-size: 0.95em; }
#footer { text-align: center; padding-top: 20px; color: #888; font-size: 0.85em; }
#footer a { color: #888; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.HTML(
        """
        <div id="header">
            <h1>📋 Consumer Complaint Classifier</h1>
            <p>Automatically routes financial consumer complaints to the correct category
            using a fine-tuned DistilBERT model, trained on CFPB complaint data.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            input_box = gr.Textbox(
                lines=8,
                placeholder="Paste or type a consumer complaint narrative here...",
                label="Complaint Narrative"
            )
            submit_btn = gr.Button("Classify Complaint", variant="primary")

            gr.Examples(
                examples=[
                    "I have been trying to dispute an incorrect entry on my credit report for months and no one responds.",
                    "A debt collector keeps calling me multiple times a day about a debt I already paid off.",
                    "My mortgage payment increased suddenly without any explanation from the lender.",
                ],
                inputs=input_box,
                label="Try an example"
            )

        with gr.Column(scale=2):
            output_status = gr.Markdown()
            output_label = gr.Label(num_top_classes=5, label="Category Probabilities")

    submit_btn.click(fn=classify_complaint, inputs=input_box, outputs=[output_label, output_status])
    input_box.submit(fn=classify_complaint, inputs=input_box, outputs=[output_label, output_status])

    gr.HTML(
        """
        <div id="footer">
            Model: <a href="https://huggingface.co/AhmedRdwan/consumer-complaint-classifier" target="_blank">DistilBERT fine-tuned on CFPB complaints</a>
            &nbsp;|&nbsp;
            <a href="https://github.com/AhmedRdwan/consumer-complaint-app" target="_blank">Source code &amp; training notebook</a>
        </div>
        """
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)