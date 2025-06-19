import torch
from embed_anything import EmbedData, ColpaliModel
import numpy as np
from pathlib import Path
import json
import logging
import os
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def create_embeddings(
    input_dir: str,
    output_dir: str,
    max_files: int = 100,
    batch_size: int = 8,
    model_name: str = "akshayballal/colpali-v1.2-merged-onnx"
) -> None:
    """
    Generate embeddings for PDF files in the input directory and save them to the output directory.
    
    Args:
        input_dir (str): Path to the directory containing PDF files.
        output_dir (str): Path to save embeddings and metadata.
        max_files (int): Maximum number of files to process.
        batch_size (int): Batch size for embedding.
        model_name (str): Name of the pretrained model.
    """
    # Initialize paths
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    logger.info(f"Loading model: {model_name}")
    try:
        model = ColpaliModel.from_pretrained_onnx(model_name)
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
    
    # Collect PDF files
    files = list(input_path.glob("*.pdf"))
    if not files:
        logger.warning(f"No PDF files found in {input_dir}")
        return
    
    logger.info(f"Found {len(files)} PDF files. Processing up to {max_files} files.")
    
    # Initialize storage for embeddings and metadata
    file_embed_data: List[EmbedData] = []
    
    # Process each file
    for idx, file in enumerate(files):
        if idx >= max_files:
            logger.info(f"Reached maximum file limit of {max_files}")
            break
            
        logger.info(f"Embedding file {idx + 1}/{min(len(files), max_files)}: {file.name}")
        try:
            embeddings: List[EmbedData] = model.embed_file(str(file), batch_size=batch_size)
            file_embed_data.extend(embeddings)
            logger.info(f"Successfully embedded {file.name} ({len(embeddings)} pages)")
        except Exception as e:
            logger.error(f"Failed to embed {file.name}: {str(e)}")
            continue
    
    if not file_embed_data:
        logger.warning("No embeddings generated")
        return
    
    # Prepare embeddings and metadata for saving
    embeddings = np.array([e.embedding for e in file_embed_data])
    metadata = [
        {
            "file_path": e.metadata.get("file_path", ""),
            "page_number": e.metadata.get("page_number", 0)
        }
        for e in file_embed_data
    ]
    
    # Save embeddings and metadata
    embeddings_file = output_path / "embeddings.npz"
    metadata_file = output_path / "metadata.json"
    
    logger.info(f"Saving embeddings to {embeddings_file}")
    np.savez(embeddings_file, embeddings=embeddings)
    
    logger.info(f"Saving metadata to {metadata_file}")
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Processed {len(file_embed_data)} pages from {min(len(files), max_files)} files")

if __name__ == "__main__":
    # Example usage
    input_directory = "data/test_repo/pdf"
    output_directory = "data/test_embeddings"
    
    create_embeddings(
        input_dir=input_directory,
        output_dir=output_directory,
        max_files=len(os.listdir(input_directory)),
        batch_size=8
    )