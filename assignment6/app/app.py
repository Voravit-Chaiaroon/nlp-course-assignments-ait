from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import os

from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.memory import ChatMessageHistory
from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain


device = torch.device("mps" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

# Load model and tokenizer
model_id = 'lmsys/fastchat-t5-3b-v1.0'
tokenizer = AutoTokenizer.from_pretrained(model_id,use_fast=False)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

prompt_template = """
    You are VoravitBot, a friendly chatbot dedicated exclusively to answering questions about Voravit's demographic and experience information. 
    Do not provide any details about yourself or your creation. If asked a question about your own age or personal attributes, 
    simply indicate that you are here to discuss Voravit's information only.
    You are Voravit, and you will respond as Voravit.  

    {context}
    Question: {question}
    Answer:
    """.strip()

PROMPT = PromptTemplate.from_template(
    template = prompt_template
)

model_name = 'hkunlp/instructor-base'
embedding_model = HuggingFaceInstructEmbeddings(
    model_name = model_name,
    model_kwargs = {"device" : device}
)

# Create a text2text-generation pipeline
pipe = pipeline(
    task="text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    model_kwargs={
        "temperature": 0,
        "repetition_penalty": 1.5,
    }
)

llm = HuggingFacePipeline(pipeline = pipe)

# A simulated function to retrieve relevant source documents
def retrieve_chain(user_query):
    
    ###Retriver###
    #calling vector from local
    vector_path = '../vector-store'
    db_file_name = 'nlp_stanford'

    vectordb = FAISS.load_local(
        folder_path = os.path.join(vector_path, db_file_name),
        embeddings = embedding_model,
        index_name = 'nlp', #default index
        allow_dangerous_deserialization=True
    )   
    retriever = vectordb.as_retriever()
    
    ###Chain###
    question_generator = LLMChain(
        llm = llm,
        prompt = CONDENSE_QUESTION_PROMPT,
        verbose = True
    )
    doc_chain = load_qa_chain(
    llm = llm,
    chain_type = 'stuff',
    prompt = PROMPT,
    verbose = True  
    )   

    ###Memory###
    memory = ConversationBufferWindowMemory(
    k=3, 
    memory_key = "chat_history",
    return_messages = True,
    output_key = 'answer'
    )

    chain = ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        return_source_documents=True,
        memory=memory,
        verbose=True,
        get_chat_history=lambda h : h
    )
    
    answer = chain({"question":user_query})
    return answer['answer'], answer['source_documents']  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message")
    # Retrieve relevant source documents
    model_response, source_documents = retrieve_chain(user_message)
    # source_docs_serializable = [
    #     {"title": doc['title'], "page_content": doc['page_content']} for doc in source_documents
    # ]
    print(source_documents)
    source_documents_formatted = [
    {
        "title": f"{doc.metadata['title']} (page {doc.metadata['page'] + 1})",
        "page_content": f"{doc.page_content}"
    }
    for doc in source_documents
]
    response_payload = {
        "response": model_response,
        "source_documents": source_documents_formatted
    }
    return jsonify(response_payload)

if __name__ == '__main__':
    app.run(debug=True)