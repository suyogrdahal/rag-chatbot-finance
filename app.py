import os
from fastapi import FastAPI
from pydantic import BaseModel

# 1. New Google GenAI Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Standard LangChain LCEL imports
# Source - https://stackoverflow.com/a
# Posted by furas, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-23, License - CC BY-SA 4.0

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()

# ---- Configuration ----
# Ensure GOOGLE_API_KEY is set in your environment variables
if "GOOGLE_API_KEY" not in os.environ:
    print("Warning: GOOGLE_API_KEY not found in environment variables.")

# ---- Build RAG Pipeline ----
# Load documents (Ensure financial_literacy.txt exists)
try:
    loader = TextLoader("testdata.txt")
    documents = loader.load()
except Exception as e:
    # Fallback if file missing (for testing)
    from langchain_core.documents import Document
    documents = [Document(page_content="Financial literacy is the ability to understand and use various financial skills.")]
    print(f"Could not load file: {e}. Using dummy data.")

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
docs = splitter.split_documents(documents)

# 2. Use Google Embeddings (text-embedding-004 is recommended for performance)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Create Vector Store
vectordb = Chroma.from_documents(docs, embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# 3. Use Gemini Chat Model
# For Gemini 3, change model to "gemini-3.0-pro" or "gemini-exp-1114" if available.
# We use 1.5-flash here as the stable, fast default.
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3
)

# 4. Create the Chain (Modern LCEL Syntax)
# We need a prompt for the "stuff" chain
prompt = ChatPromptTemplate.from_template("""
Answer the user's question based on the context provided below:
<context>
{context}
</context>

Question: {input}
""")

document_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, document_chain)

# ---- API Schema ----
class Question(BaseModel):
    query: str

@app.post("/ask")
def ask_question(item: Question):
    # invoke returns a dictionary with 'answer' and 'context'
    response = qa_chain.invoke({"input": item.query})
    return {"answer": response["answer"]}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Financial Literacy RAG Chatbot API (Powered by Gemini)!"}