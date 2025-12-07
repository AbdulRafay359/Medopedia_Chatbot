from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


load_dotenv()

os.environ["PINECONE_API_KEY"] = "pcsk_5b31iK_PEauo39XK6uigdNvXRiGLpzZKXY6DMgjyMyquXCWYVPognBmAGqCS27LNxsAz1t"

extracted_data = load_pdf_file(data='Data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key="pcsk_5b31iK_PEauo39XK6uigdNvXRiGLpzZKXY6DMgjyMyquXCWYVPognBmAGqCS27LNxsAz1t")
index_name = "medicalchatrobot"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

vectorstore = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)

