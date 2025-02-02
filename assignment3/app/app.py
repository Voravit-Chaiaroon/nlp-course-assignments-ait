from flask import Flask, render_template, request, jsonify
import torch
from Seq2SeqTransformer import Seq2SeqTransformer
from Seq2SeqTransformer import Encoder
from Seq2SeqTransformer import Decoder
from Seq2SeqTransformer import EncoderLayer
from Seq2SeqTransformer import DecoderLayer
from Seq2SeqTransformer import MultiHeadAttentionLayer
from Seq2SeqTransformer import PositionwiseFeedforwardLayer
from torchtext.data.utils import get_tokenizer
from pythainlp.tokenize import word_tokenize  # Thai tokenizer
from flask_cors import CORS
import sys

app = Flask(__name__)
CORS(app)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/Seq2SeqTransformergeneralAttention.pt"

# Load Vocab
vocab = torch.load("models/vocab_transform.pth")

# Define hyperparameters
input_dim   = 16902
output_dim  = 12786
hid_dim = 256
enc_layers = 3
dec_layers = 3
enc_heads = 8
dec_heads = 8
enc_pf_dim = 512
dec_pf_dim = 512
enc_dropout = 0.1
dec_dropout = 0.1
src_pad_idx = 1  # Padding index for source language
trg_pad_idx = 1  # Padding index for target language
attention_type = "general"  # Choose from "general", "additive", "scaled_dot", "multiplicative"

# Instantiate the encoder and decoder
encoder = Encoder(input_dim, hid_dim, enc_layers, enc_heads, enc_pf_dim, enc_dropout, device, attention_type)
decoder = Decoder(output_dim, hid_dim, dec_layers, dec_heads, dec_pf_dim, dec_dropout, device, attention_type)

# Instantiate the Seq2Seq model
model = Seq2SeqTransformer(encoder, decoder, src_pad_idx, trg_pad_idx, device)
model.load_state_dict(torch.load(model_path))
model.eval()
vocab = torch.load("models/vocab_transform.pth")

# Define special symbols and indices
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

# Place-holders
token_transform = {}
vocab_transform = {}
SRC_LANGUAGE = 'en'
TRG_LANGUAGE = 'th'

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
token_transform[TRG_LANGUAGE] = word_tokenize

vocab_transform = vocab

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids):
    return torch.cat((torch.tensor([SOS_IDX]), 
                      torch.tensor(token_ids), 
                      torch.tensor([EOS_IDX])))


from torchtext.data.utils import get_tokenizer
from pythainlp.tokenize import word_tokenize  # Thai tokenizer

# src and trg language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor

def translate(text):
    model.eval()
    src_text = text_transform[SRC_LANGUAGE](text).to(device)
    trg_text = text_transform[TRG_LANGUAGE](text).to(device)
    src_text = src_text.reshape(-1, 1)  
    trg_text = trg_text.reshape(-1, 1)
    text_length = torch.tensor([src_text.size(0)]).to(dtype=torch.int64)
    with torch.no_grad():
        output, attentions = model(src_text, text_length, trg_text, 0) #turn off teacher forcing
    
    output = output.squeeze(1)
    output = output[1:]
    output_max = output.argmax(1)
    translation = ''

    for token in output_max:
        translation = translation + mapping[token.item()]
    mapping = vocab_transform[TRG_LANGUAGE].get_itos()

    translation = ''
    for token in output_max:
        translation = translation + mapping[token.item()]
    if not text:
        return jsonify({"error": "Word is required"}), 400

    return jsonify({
        "translatedText": translation
    })

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translation():
    try:
        data = request.json
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        print(f"Received input: {text}")  # Debug log

        return translate(text)
    
    except Exception as e:
        print(f"Error: {e}")  # Print exact error
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)