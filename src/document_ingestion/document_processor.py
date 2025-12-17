"""Document processing module for loading and splitting documents"""

from typing import List, Union
from pathlib import Path

from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    PyPDFDirectoryLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentProcessor:
    """Handles document loading and processing"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # -------- Loaders -------- #

    def load_from_url(self, url: str) -> List[Document]:
        return WebBaseLoader(url).load()

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        return PyPDFLoader(str(file_path)).load()

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        return PyPDFDirectoryLoader(str(directory)).load()

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        return TextLoader(str(file_path), encoding="utf-8").load()

    # -------- Dispatcher -------- #

    def load_documents(self, sources: List[str]) -> List[Document]:
        docs: List[Document] = []

        for src in sources:
            if src.startswith(("http://", "https://")):
                docs.extend(self.load_from_url(src))
                continue

            path = Path(src)

            if path.is_dir():
                docs.extend(self.load_from_pdf_dir(path))
            elif path.suffix.lower() == ".pdf":
                docs.extend(self.load_from_pdf(path))
            elif path.suffix.lower() == ".txt":
                docs.extend(self.load_from_txt(path))
            else:
                raise ValueError(f"Unsupported source type: {src}")

        return docs

    # -------- Splitter -------- #

    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)

    def process_sources(self, sources: List[str]) -> List[Document]:
        docs = self.load_documents(sources)
        return self.split_documents(docs)
