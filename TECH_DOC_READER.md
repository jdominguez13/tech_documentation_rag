## PROJECT: Local AI Assistant for Technical Documentation Q&A (LangChain-Based)

### GOAL:
Build a local AI chatbot using LangChain and Streamlit that allows users to upload technical documents or provide URLs (e.g., PDFs, DOCX, Markdown, Swagger/OpenAPI, network/security policies), ask questions in natural language, and receive answers grounded in the uploaded materials. The chatbot should:
- Ask for more information when the documentation is insufficient
- Make best-effort suggestions with a disclaimer if going beyond source material
- Allow users to paste error messages and suggest fixes
- Include source attribution for transparency

---

### ARCHITECTURE:

#### 🔧 Tools & Libraries:
- **Frontend**: Streamlit (browser-based chat UI)
- **RAG Framework**: LangChain (retrieval-augmented generation pipeline)
- **Vector Store**: Local ChromaDB instance (via `langchain.vectorstores.Chroma`)
- **Embeddings**: OpenAI (`text-embedding-3-small`)
- **LLM**: OpenAI GPT-4 or GPT-4o (via LangChain `ChatOpenAI`)
- **Document Loaders**:
  - PDFs: `PyMuPDF` or `pdfplumber`
  - DOCX: `python-docx`
  - Markdown: `markdown` or plain text
  - Swagger/OpenAPI: parse YAML/JSON → flatten to human-readable Markdown
  - URLs: `requests` + `BeautifulSoup`

---

### ✅ FUNCTIONAL REQUIREMENTS:

- **Upload Support** for PDFs, DOCX, Markdown, Swagger specs
- **URL Support** for loading web documentation (text extraction from HTML)
- Normalize all documents to **Markdown** format
- Use LangChain’s `Document` objects for chunking and metadata
- Store embeddings in ChromaDB with document tags for source tracking
- Use retrieval with metadata filtering to fetch top relevant chunks
- Generate answers via GPT-4 with prompt engineering for:
  - Clarification questions if unsure
  - Disclaimers for best guesses
  - Debugging suggestions from pasted error messages
- Display answer + sources in Streamlit chat interface

---

### ✅ FILE STRUCTURE:

ai-docs-bot/
├── main.py # Streamlit app + LangChain pipeline
├── loaders/
│ ├── pdf_loader.py
│ ├── docx_loader.py
│ ├── url_loader.py
│ ├── swagger_loader.py
├── vector_store/ # ChromaDB files
├── docs/ # Uploaded source documents
├── prompts/
│ └── base_prompt.txt # Prompt template with safety rules
├── CLOUD_EXPANSION_PLAN.md # Guide for cloud-deployable version
└── README.md # Setup + usage instructions