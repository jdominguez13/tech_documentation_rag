import yaml
import json
from langchain.schema import Document
import os

def load_swagger(file_path, source_name=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.json'):
            data = json.load(f)
        else:
            data = yaml.safe_load(f)
    # Flatten to Markdown (stub)
    text = str(data)
    metadata = {"source": source_name or os.path.basename(file_path)}
    return [Document(page_content=text, metadata=metadata)] 