import streamlit as st
import os
import tempfile
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

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
PROMPT_PATH = "prompts/base_prompt.txt"

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
url_input = st.sidebar.text_input("Or paste a documentation URL")

st.markdown("---")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None
if 'docs_loaded' not in st.session_state:
    st.session_state['docs_loaded'] = False

# Helper: Load prompt
def load_prompt():
    with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
        return f.read()

# Helper: Load and normalize documents
def load_documents(files, url):
    docs = []
    for file in files:
        filename = file.name
        ext = filename.lower().split('.')[-1]
        # Save file to docs dir
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
    if url:
        docs.extend(url_loader.load_url(url))
    return docs

# Helper: Chunk documents
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# Helper: Build vector store
def build_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        collection_name="docs",
        embedding_function=embeddings,
        persist_directory=VECTOR_STORE_DIR
    )
    vectordb.add_documents(chunks)
    vectordb.persist()
    return vectordb

# Ingest documents if uploaded or URL provided
if (uploaded_files or url_input) and not st.session_state['docs_loaded']:
    with st.spinner("Processing documents and building knowledge base..."):
        docs = load_documents(uploaded_files, url_input)
        if not docs:
            st.error("No valid documents found.")
        else:
            chunks = chunk_documents(docs)
            vectordb = build_vector_store(chunks)
            st.session_state['vector_store'] = vectordb
            st.session_state['docs_loaded'] = True
            st.success(f"Loaded {len(docs)} document(s), {len(chunks)} chunks indexed.")

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
        st.error("Please upload documents or provide a URL first.")
    else:
        with st.spinner("Retrieving answer..."):
            retriever = st.session_state['vector_store'].as_retriever(search_kwargs={"k": 4})
            prompt = load_prompt()
            llm = ChatOpenAI(temperature=0, model_name="gpt-4")
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
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