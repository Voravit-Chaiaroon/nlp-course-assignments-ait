from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertModel, BertConfig

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer and custom model.
# For demonstration, we load a Hugging Face model.
# Replace with your custom paths and loading code as needed.
MODEL_PATH = "./models/s_bert_model.pth"

# Example: load the tokenizer (if using BertTokenizer)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the custom trained model.

# Here we assume a three-class (entailment, neutral, contradiction) classifier.
model = BertModel.from_pretrained(
    'bert-base-uncased',
    num_labels=3,
    output_hidden_states=False,
    output_attentions=False
)

# Create a configuration with custom dimensions
config = BertConfig.from_pretrained('bert-base-uncased')
config.vocab_size = 60305              # Match your checkpoint's vocab size
config.max_position_embeddings = 1000  # Match your checkpoint's max positions
config.hidden_size = 768               # Match your checkpoint's hidden size
config.num_attention_heads = 12        # Match your checkpoint's number of attention heads
config.intermediate_size = 3072        # Match your checkpoint's intermediate size
config.hidden_dropout_prob = 0.1       # Match your checkpoint's hidden dropout probability
# Instantiate the model with the updated configuration
model = BertModel(config)

# Load your custom state dict.
state_dict = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(state_dict, strict=False)
model.eval()
classifier_head = torch.nn.Linear(768*3, 3).to(device)
classifier_head.eval()

# Define a mapping from prediction index to label.
label_map = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

# define mean pooling function
def mean_pool(token_embeds, attention_mask):
    # reshape attention_mask to cover 768-dimension embeddings
    in_mask = attention_mask.unsqueeze(-1).expand(
        token_embeds.size()
    ).float()
    # perform mean-pooling but exclude padding tokens (specified by in_mask)
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-9
    )
    return pool

def predict_nli(premise: str, hypothesis: str) -> str:
    """
    Predict the NLI label for the given premise and hypothesis.
    """
    # Tokenize separately for premise and hypothesis:
    inputs_ids_a = tokenizer(premise,
                            return_tensors="pt",
                            truncation=True,
                            padding=True)
    inputs_ids_b = tokenizer(hypothesis,
                            return_tensors="pt",
                            truncation=True,
                            padding=True)
    
    with torch.no_grad():
        inputs_ids_a = {k: v.to(device) for k, v in inputs_ids_a.items()}
        inputs_ids_b = {k: v.to(device) for k, v in inputs_ids_b.items()}

        # Forward pass
        u = model(**inputs_ids_a)  
        v = model(**inputs_ids_b)  

        u_last_hidden_state = u.last_hidden_state
        v_last_hidden_state = v.last_hidden_state

        # Mean pooling
        u_mean_pool = mean_pool(u_last_hidden_state, inputs_ids_a['attention_mask'])
        v_mean_pool = mean_pool(v_last_hidden_state, inputs_ids_b['attention_mask'])
        
        uv_abs = torch.abs(u_mean_pool - v_mean_pool)
        x = torch.cat([u_mean_pool, v_mean_pool, uv_abs], dim=-1)

        # Get predictions
        logits = classifier_head(x)
        probabilities = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probabilities, dim=1).item()
    
    return label_map[pred_class]

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        premise = request.form.get('premise', '')
        hypothesis = request.form.get('hypothesis', '')
        if premise and hypothesis:
            prediction = predict_nli(premise, hypothesis)
            result = f"Prediction: {prediction}"
        else:
            result = "Please enter both a premise and a hypothesis."
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
