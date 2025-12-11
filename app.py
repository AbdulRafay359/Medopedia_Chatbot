from src.helper import download_hugging_face_embeddings
from flask import Flask, render_template, jsonify, request
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from src.prompt import *
import os

app = Flask(__name__)


load_dotenv()

os.environ["PINECONE_API_KEY"] = "pcsk_5b31iK_PEauo39XK6uigdNvXRiGLpzZKXY6DMgjyMyquXCWYVPognBmAGqCS27LNxsAz1t"

embeddings = download_hugging_face_embeddings()

index_name = "medicalchatrobot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

llm = OllamaLLM(model="llama3.1")

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

def rag_answer(query):
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])

    formatted_prompt = prompt.format(
        context=context,
        input=query
    )

    response = llm.invoke(formatted_prompt)

    return response


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User:", msg)

    response = rag_answer(msg)
    
    return str(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)