from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import tempfile
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
#from langchain_community.chat_models import ChatTogether
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Import loaders
def import_loader(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, f"loaders/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

pdf_loader = import_loader("pdf_loader")
docx_loader = import_loader("docx_loader")
url_loader = import_loader("url_loader")
swagger_loader = import_loader("swagger_loader")

# Constants
VECTOR_STORE_DIR = "vector_store"
DOCS_DIR = "docs"
PROMPT_PATH = "prompts/base_prompt_math_docs.txt"
# For Will's Example
#PROMPT_PATH = "prompts/base_prompt.txt"

os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

st.set_page_config(page_title="AI Docs Bot", layout="wide")
st.title("📄 AI Technical Docs Q&A Bot")

st.sidebar.header("Upload or Link Documentation")
uploaded_files = st.sidebar.file_uploader(
    "Upload files (PDF, DOCX, Markdown, Swagger)",
    type=["pdf", "docx", "md", "yaml", "yml", "json"],
    accept_multiple_files=True
)
url_input = st.sidebar.text_area(
    "Paste one or more documentation URLs (one per line or comma-separated)",
    height=100
)
add_docs = st.sidebar.button("Add Documents/URLs")

st.markdown("---")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None
if 'docs_loaded' not in st.session_state:
    st.session_state['docs_loaded'] = False
if 'added_sources' not in st.session_state:
    st.session_state['added_sources'] = set()

def parse_urls(url_input):
    if not url_input:
        return []
    # Split by newlines or commas, strip whitespace
    urls = [u.strip() for u in url_input.replace(',', '\n').split('\n') if u.strip()]
    return urls

# Helper: Load prompt
def load_prompt():
    with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
        return f.read()

# Helper: Load and normalize documents
def load_documents(files, urls):
    docs = []
    # Only process new files/URLs
    for file in files:
        filename = file.name
        if filename in st.session_state['added_sources']:
            continue
        ext = filename.lower().split('.')[-1]
        file_path = os.path.join(DOCS_DIR, filename)
        with open(file_path, 'wb') as f:
            f.write(file.read())
        if ext == 'pdf':
            docs.extend(pdf_loader.load_pdf(file_path, filename))
        elif ext == 'docx':
            docs.extend(docx_loader.load_docx(file_path, filename))
        elif ext in ['yaml', 'yml', 'json']:
            docs.extend(swagger_loader.load_swagger(file_path, filename))
        elif ext == 'md':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            docs.append(Document(page_content=text, metadata={"source": filename}))
        st.session_state['added_sources'].add(filename)
    for url in urls:
        if url in st.session_state['added_sources']:
            continue
        docs.extend(url_loader.load_url(url))
        st.session_state['added_sources'].add(url)
    return docs

# Helper: Chunk documents
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# Helper: Build or update vector store
def add_to_vector_store(chunks):
    if st.session_state['vector_store'] is None:
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma(
            collection_name="docs",
            embedding_function=embeddings,
            persist_directory=VECTOR_STORE_DIR
        )
        vectordb.add_documents(chunks)
        vectordb.persist()
        st.session_state['vector_store'] = vectordb
    else:
        st.session_state['vector_store'].add_documents(chunks)
        st.session_state['vector_store'].persist()

# Add new documents/URLs when button is clicked
if add_docs:
    urls = parse_urls(url_input)
    docs = load_documents(uploaded_files, urls)
    if not docs:
        st.warning("No new documents or URLs to add.")
    else:
        with st.spinner("Processing and adding new documents/URLs to knowledge base..."):
            chunks = chunk_documents(docs)
            add_to_vector_store(chunks)
            st.session_state['docs_loaded'] = True
            st.success(f"Added {len(docs)} new document(s)/URL(s), {len(chunks)} chunks indexed.")

# Load vector store if already built
if st.session_state['docs_loaded'] and st.session_state['vector_store'] is None:
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        collection_name="docs",
        embedding_function=embeddings,
        persist_directory=VECTOR_STORE_DIR
    )
    st.session_state['vector_store'] = vectordb

# Q&A Section
question = st.text_input("Ask a question about your documentation:")

if st.button("Ask") and question:
    if not st.session_state['vector_store']:
        st.error("Please add documents or URLs first.")
    else:
        with st.spinner("Retrieving answer..."):
            retriever = st.session_state['vector_store'].as_retriever(search_kwargs={"k": 4})
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template=load_prompt()
            )
            #llm = ChatOpenAI(temperature=0, model_name="gpt-4")
            llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
            #llm = ChatTogether(model="meta-llama/Llama-3-70b-chat-hf", together_api_key="YOUR_KEY")
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt_template}
            )
            result = qa({"query": question})
            answer = result["result"]
            sources = []
            for doc in result.get("source_documents", []):
                src = doc.metadata.get("source", "Unknown")
                if src not in sources:
                    sources.append(src)
            st.session_state['chat_history'].append((question, answer, sources))

# Display chat history
for q, a, s in st.session_state['chat_history']:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
    st.markdown(f"_Sources: {', '.join(s)}_")
    st.markdown("---") 