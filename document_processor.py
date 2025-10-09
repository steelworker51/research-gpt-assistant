# document_processor.py
from pathlib import Path
from typing import List, Tuple, Dict
import re
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DocumentProcessor:
    def __init__(self, config):
        #1. self.docs (a list of chunk texts) and self.doc_meta (a list of metadata dictionaries, one per chunk) for simplified storage.
        #2. self.vectorizer is now fully initialized with common NLP parameters (stop_words, ngram_range, max_features), moving from a placeholder (None)
        #   to an operational component.
        self.config = config
        self.docs: List[str] = []
        self.doc_meta: List[Dict] = []  # [{"path": ..., "chunk_id": ...}, ...]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=50_000,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None

    # --- High priority ---
    # Uses PdfReader (from PyPDF2) to iterate through all pages, extracts text, handles potential None return from extract_text(), and joins pages with a newline.
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        reader = PdfReader(str(pdf_path)) #type hinting (pdf_path: Path -> str)
        texts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            texts.append(t)
        return "\n".join(texts)

    def preprocess_text(self, text: str) -> str: #: 1. Removes null characters (\x00). 2. Replaces all whitespace sequences with a single space (\s+ to ). 
        #3. De-hyphenates words broken by line breaks (-\s+ to ""). 4. Strips leading/trailing whitespace. type hinting (text: str -> str).
        # Basic cleaning; keep it simple and fast
        text = text.replace("\x00", " ")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"-\s+", "", text)  # de-hyphenate line breaks
        return text.strip()

    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        # while loop and string slicing. The implementation drops the reliance on self.config parameters within the function definition 
        # (they are passed in by the caller, add_pdf) and the complexity of not breaking sentences, prioritizing simplicity and speed.
        chunks = []
        i = 0
        while i < len(text):
            chunk = text[i:i+chunk_size]
            chunks.append(chunk)
            i += chunk_size - overlap
        return chunks

    # --- Indexing ---
    def add_pdf(self, pdf_path: Path): # 1. Extracts, preprocesses, and chunks text from a PDF file. 2. Appends each chunk to self.docs and its metadata 
        #(file path and chunk ID) to self.doc_meta.
        raw = self.extract_text_from_pdf(pdf_path)
        cleaned = self.preprocess_text(raw)
        chunks = self.chunk_text(cleaned, self.config.chunk_size, self.config.chunk_overlap)

        for idx, ch in enumerate(chunks): # Store chunk and metadata
            self.docs.append(ch)
            self.doc_meta.append({"path": str(pdf_path), "chunk_id": idx})

    def build_search_index(self): # 1. Validates that documents have been added (raises ValueError if not). 
        # 2. Fits the TF-IDF vectorizer on self.docs and transforms them into a TF-IDF matrix.
        if not self.docs:
            raise ValueError("No documents added. Add PDFs before building the index.")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.docs)

    def find_similar_chunks(self, query: str, top_k: int = 5) -> List[Tuple[float, str, Dict]]: # 1. Validates that the search index has been built (raises ValueError if not).
        if self.tfidf_matrix is None: # Check if the TF-IDF matrix is built
            raise ValueError("Index not built. Call build_search_index() first.")
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.tfidf_matrix).ravel()
        top_idx = sims.argsort()[::-1][:top_k]
        results = []
        for i in top_idx:
            results.append((float(sims[i]), self.docs[i], self.doc_meta[i]))
        return results
