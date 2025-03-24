from flask import Flask, render_template, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = Flask(__name__)

# Define the path to your model and tokenizer
model_path = "models"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    local_files_only=True  # Ensures it loads from local files
)

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def classify_text(input_text):
    """Predict whether the input text is toxic or not."""
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move input tensors to the correct device

    with torch.no_grad():
        outputs = model(**inputs)
    
    # Assuming the model outputs logits
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label = torch.argmax(predictions, dim=1).item()

    # Define class labels (modify if different)
    class_labels = ["Non-Toxic", "Toxic"]
    return class_labels[label]


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    input_text = ""
    if request.method == "POST":
        input_text = request.form["text"]
        # Classify the input text.
        prediction = classify_text(input_text)
        
        if prediction == "Toxic":
            result = f"The text is classified as TOXIC."
        else: #Non-Toxic
            result = f"The text is NOT classified as toxic."
    return render_template("index.html", result=result, input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)
