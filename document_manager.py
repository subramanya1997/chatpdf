import os
from pathlib import Path
import logging

class DocumentManager:
    """
    Manager class for loading and accessing document files from a specified directory.
    Similar to PromptManager but designed to work with Markdown files or other document formats.
    """
    def __init__(self, docs_dir="mdfiles"):
        """
        Initialize the DocumentManager with the specified directory.
        
        Args:
            docs_dir (str): Path to the directory containing document files
        """
        self.docs_dir = docs_dir
        self.documents = {}
        self.load_documents()
    
    def load_documents(self):
        """Load all document files from the documents directory"""
        docs_path = Path(self.docs_dir)
        if not docs_path.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.docs_dir}")
        
        # Load all markdown files and other document types
        for doc_file in docs_path.glob("*.md"):
            file_name = doc_file.name  # Include extension in the key
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    self.documents[file_name] = f.read()
            except Exception as e:
                print(f"Error loading document {file_name}: {e}")
    
    def get_document(self, file_name):
        """
        Get a specific document by filename (including extension)
        
        Args:
            file_name (str): The filename with extension (e.g., 'document.md')
            
        Returns:
            str: The content of the document
            
        Raises:
            KeyError: If the document is not found
        """
        if file_name not in self.documents:
            raise KeyError(f"Document not found: {file_name}")
        return self.documents[file_name]
    
    def get_document_names(self):
        """
        Get a list of all available document names
        
        Returns:
            list: List of document filenames
        """
        return list(self.documents.keys())
    
    def refresh_documents(self):
        """Reload all documents from disk"""
        self.documents.clear()
        self.load_documents()
