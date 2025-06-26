import fitz  # PyMuPDF
from langchain.schema import Document
import os

def load_pdf(file_path, source_name=None):
    doc = fitz.open(file_path)
    text = "\n".join(page.get_text() for page in doc)
    # Optionally convert to Markdown here (stub)
    metadata = {"source": source_name or os.path.basename(file_path)}
    return [Document(page_content=text, metadata=metadata)] 