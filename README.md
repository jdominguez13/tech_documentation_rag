# Local AI Assistant for Technical Documentation Q&A

This project is a local AI chatbot that answers questions about your technical documentation using LangChain and Streamlit. Upload documents (PDF, DOCX, Markdown, Swagger/OpenAPI, or URLs), ask questions, and get answers grounded in your docs—with source attribution and clarifications when needed.

## Features
- Upload PDFs, DOCX, Markdown, Swagger/OpenAPI specs
- Paste URLs to load web documentation
- Paste error messages for debugging suggestions
- Answers always reference sources
- Runs locally (no cloud required)

## Setup
1. Clone this repo and `cd ai-docs-bot`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY=your-key-here
   ```
4. Run the app:
   ```bash
   streamlit run main.py
   ```

## Usage
- Upload your docs or paste a URL
- Ask questions in the chat
- Get answers with source references

## File Structure
- `main.py`: Streamlit app + LangChain pipeline
- `loaders/`: Document loaders for each file type
- `vector_store/`: Local ChromaDB files
- `docs/`: Uploaded source documents
- `prompts/base_prompt.txt`: Prompt template

---
See `TECH_DOC_READER.md` for full design details. 