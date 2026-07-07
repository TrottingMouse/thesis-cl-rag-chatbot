import pickle
from pathlib import Path

import faiss
import torch
from sentence_transformers import SentenceTransformer

from src.models import Chunk
from .base import BaseIndexBuilder


class FaissIndexBuilder(BaseIndexBuilder):
    """
    Builds and manages a FAISS-based dense vector index.
    """

    def __init__(self, storage_path: Path, model_name: str):
        """
        Parameters
        ----------
        storage_path:
            Directory under which the index artefacts are stored.
        model_name:
            Name of the embedding model to use (e.g., from sentence-transformers).
        """
        super().__init__(storage_path)
        self.model_name = model_name
        
        # Automatically detect GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trust_remote_code = "jina" in model_name
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=trust_remote_code)
        
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    @property
    def name(self) -> str:
        return "faiss_" + self.model_name.replace("/", "_")

    def build(self, chunks: list[Chunk]) -> None:
        """
        Build the FAISS index from a list of chunks and persist it to disk.
        
        Parameters
        ----------
        chunks:
            All chunks to be indexed. The builder embeds them and persists the result.
        """
        if not chunks:
            raise ValueError("Cannot build index with an empty chunks list.")

        self.chunks = chunks
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Extract text from chunks
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True,
            show_progress_bar=True, 
            task='retrieval', 
            prompt_name="document")
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        
        # We use IndexFlatIP (Inner Product) to compute Cosine Similarity, 
        # which requires L2-normalizing the embeddings first.
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(dimension)
        
        # Add vectors to the index
        self.index.add(embeddings)
        
        # Save FAISS index
        index_path = self.storage_path / "index.faiss"
        faiss.write_index(self.index, str(index_path))
        
        # Save chunks metadata mapping
        chunks_path = self.storage_path / "chunks.pkl"
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

    # def load(self) -> None:
    #     """
    #     Load a previously built index and chunks from disk into memory.
    #     """
    #     if not self.is_built():
    #         raise FileNotFoundError(f"FAISS index not found at {self.storage_path}")

    #     index_path = self.storage_path / "index.faiss"
    #     self.index = faiss.read_index(str(index_path))

    #     chunks_path = self.storage_path / "chunks.pkl"
    #     with open(chunks_path, "rb") as f:# Uses the overridden value!
    #         self.chunks = pickle.load(f)

    # def is_built(self) -> bool:
    #     """
    #     Return True if both the faiss index and chunks file exist.
    #     """
    #     index_path = self.storage_path / "index.faiss"
    #     chunks_path = self.storage_path / "chunks.pkl"
    #     return index_path.exists() and chunks_path.exists()