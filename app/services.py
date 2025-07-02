import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from transformers.utils.import_utils import is_flash_attn_2_available
from pathlib import Path
import json
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DocumentRetriever:
    def __init__(self, embeddings_dir: str, model_name: str = "vidore/colqwen2-v1.0"):
        self.embeddings_dir = Path(embeddings_dir)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing model on {self.device}")
        self.model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(model_name)
        self.embeddings: torch.Tensor = None
        self.metadata: List[Dict] = []
        self.load_embeddings()

    def load_embeddings(self):
        """
        Load embeddings and metadata from files.
        """
        embeddings_file = self.embeddings_dir / "image_embeddings.pt"
        metadata_file = self.embeddings_dir / "image_metadata.json"

        if not embeddings_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"Embeddings or metadata not found in {self.embeddings_dir}")

        logger.info(f"Loading embeddings from {embeddings_file}")
        self.embeddings = torch.load(embeddings_file, map_location="cpu")

        logger.info(f"Loading metadata from {metadata_file}")
        with metadata_file.open("r") as f:
            self.metadata = json.load(f)

        if len(self.metadata) != self.embeddings.size(0):
            raise ValueError("Mismatch between embeddings and metadata lengths")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for documents matching the query.
        Returns top_k results with file paths and page numbers.
        """
        if not self.metadata or self.embeddings is None:
            raise ValueError("No embeddings loaded")

        logger.info(f"Processing query: {query}")
        
        query_inputs = self.processor.process_queries([query]).to(self.device)
        with torch.no_grad():
            query_embeddings = self.model(**query_inputs)

        # Compute scores
        scores = self.processor.score_multi_vector(query_embeddings, self.embeddings.to(self.device))
        top_indices = torch.topk(scores[0], k=top_k).indices.cpu().numpy()

        for i, query in enumerate([query]):
            print(f"\nTop results for query: {query}")
            top_indices = torch.topk(scores[i], k=top_k).indices
            for rank, idx in enumerate(top_indices):
                print(f"{rank + 1}. {self.metadata[idx]}")

        return [
            {
                "file_path": self.metadata[idx]["file_path"],
                "page_number": self.metadata[idx]["page_number"]
            }
            for idx in top_indices
        ]

    def cleanup(self):
        """
        Clean up resources.
        """
        self.metadata.clear()
        self.embeddings = None