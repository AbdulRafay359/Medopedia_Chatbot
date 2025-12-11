from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
from src.prompt import *
import os

app = FastAPI()

templates = Jinja2Templates(directory="templates")  # put chat.html inside templates/
app.mount("/static", StaticFiles(directory="static"), name="static")

load_dotenv()
os.environ["PINECONE_API_KEY"] = "ADD_YOUR_PINECONE_API_KEY"

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


def rag_answer(query: str):
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])

    formatted_prompt = prompt.format(
        context=context,
        input=query
    )

    response = llm.invoke(formatted_prompt)
    return response



@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/get")
async def chat(msg: str = Form(...)):
    print("User:", msg)
    response = rag_answer(msg)
    return {"response": response}

# -------------------------
# Run server
# -------------------------
# Use: uvicorn main:app --reload --host 0.0.0.0 --port 8080
