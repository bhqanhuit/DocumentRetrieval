import torch
from embed_anything import EmbedData, ColpaliModel
import numpy as np
from pathlib import Path
import json
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DocumentRetriever:
    def __init__(self, embeddings_dir: str):
        self.model = ColpaliModel.from_pretrained_onnx("akshayballal/colpali-v1.2-merged-onnx")
        self.embeddings_dir = Path(embeddings_dir)
        self.file_embed_data: List[Dict] = []
        self.embeddings: np.ndarray = None
        self.load_embeddings()

    def load_embeddings(self):
        """
        Load embeddings and metadata from files.
        """
        embeddings_file = self.embeddings_dir / "embeddings.npz"
        metadata_file = self.embeddings_dir / "metadata.json"

        if not embeddings_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"Embeddings or metadata not found in {self.embeddings_dir}")

        logger.info(f"Loading embeddings from {embeddings_file}")
        with np.load(embeddings_file) as data:
            self.embeddings = data["embeddings"]

        logger.info(f"Loading metadata from {metadata_file}")
        with metadata_file.open("r") as f:
            self.file_embed_data = json.load(f)

        if len(self.file_embed_data) != len(self.embeddings):
            raise ValueError("Mismatch between embeddings and metadata lengths")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for documents matching the query.
        Returns top_k results with file paths and page numbers.
        """
        if not self.file_embed_data or self.embeddings is None:
            raise ValueError("No embeddings loaded")

        logger.info(f"Processing query: {query}")
        query_embedding = self.model.embed_query(query)
        query_embeddings = np.array([e.embedding for e in query_embedding])

        # Compute similarity scores
        scores = np.einsum("bnd,csd->bcns", query_embeddings, self.embeddings).max(axis=3).sum(axis=2).squeeze()
        top_pages = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "file_path": self.file_embed_data[page]["file_path"],
                "page_number": self.file_embed_data[page]["page_number"]
            }
            for page in top_pages
        ]

    def cleanup(self):
        """
        Clean up resources.
        """
        self.file_embed_data.clear()
        self.embeddings = None