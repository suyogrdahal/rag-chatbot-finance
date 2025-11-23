from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAI

app = FastAPI()

# ---- Build RAG Pipeline ----
loader = TextLoader("financial_literacy.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
docs = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(docs, embeddings)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini")

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# ---- API Schema ----
class Question(BaseModel):
    query: str

@app.post("/ask")
def ask_question(item: Question):
    answer = qa_chain.run(item.query)
    return {"answer": answer}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Financial Literacy RAG Chatbot API!"}