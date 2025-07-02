
import torch
from pathlib import Path
import json
import logging
from typing import List, Dict, Any
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def consolidate_embeddings(
    input_dir: str = "data/preprocessed_documents",
    output_dir: str = "data/doc_embedding",
    metadata_input_path: str = "data/doc_embedding/image_metadata.json",
    embeddings_output_path: str = "data/doc_embedding/image_embeddings.pt",
    metadata_output_path: str = "data/doc_embedding/image_metadata.json",
    expected_shape: tuple = (1, 75, 128)
) -> None:
    """
    Load individual embedding files, stack them, and save as a single tensor.
    Also copy or update metadata to the output directory.

    Args:
        input_dir (str): Directory containing individual embedding (.pt) and image files.
        output_dir (str): Directory to save the consolidated embeddings and metadata.
        metadata_input_path (str): Path to the input metadata (.json) file.
        embeddings_output_path (str): Path to save the consolidated embeddings (.pt) file.
        metadata_output_path (str): Path to save the output metadata (.json) file.
        expected_shape (tuple): Expected shape of each embedding (batch_size, height, width).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load metadata
    try:
        with open(metadata_input_path, "r") as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata from {metadata_input_path} with {len(metadata)} entries")
    except Exception as e:
        logger.error(f"Failed to load metadata from {metadata_input_path}: {str(e)}")
        return

    if not metadata:
        logger.warning("No metadata found")
        return

    # Load and validate individual embeddings
    embeddings = []
    valid_metadata = []
    for item in tqdm(metadata, desc="Loading embeddings"):
        embedding_file = item.get("embedding_file")
        if not embedding_file:
            logger.warning(f"Missing embedding_file in metadata for {item.get('image_file', 'unknown')}")
            continue

        embedding_path = input_path / embedding_file
        try:
            embedding = torch.load(embedding_path, map_location="cpu")
            if embedding.shape == expected_shape:
                embeddings.append(embedding)
                valid_metadata.append(item)
                logger.info(f"Loaded embedding from {embedding_path} with shape {embedding.shape}")
            else:
                logger.warning(
                    f"Embedding at {embedding_path} has unexpected shape {embedding.shape}, "
                    f"expected {expected_shape}. Skipping."
                )
        except Exception as e:
            logger.error(f"Failed to load embedding from {embedding_path}: {str(e)}. Skipping.")
            continue

    if not embeddings:
        logger.error("No valid embeddings loaded")
        return

    # Stack embeddings
    try:
        stacked_embeddings = torch.cat(embeddings, dim=0)  # Shape: [num_embeddings, 75, 128]
        logger.info(f"Stacked embeddings shape: {stacked_embeddings.shape}")
    except Exception as e:
        logger.error(f"Failed to stack embeddings: {str(e)}")
        return

    # Save consolidated embeddings
    try:
        torch.save(stacked_embeddings, embeddings_output_path)
        logger.info(f"Saved consolidated embeddings to {embeddings_output_path}")
    except Exception as e:
        logger.error(f"Failed to save consolidated embeddings to {embeddings_output_path}: {str(e)}")
        return

    # Save metadata (copy or update paths if needed)
    try:
        with open(metadata_output_path, "w") as f:
            json.dump(valid_metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_output_path}")
    except Exception as e:
        logger.error(f"Failed to save metadata to {metadata_output_path}: {str(e)}")
        return

    logger.info(f"Successfully consolidated {len(embeddings)} embeddings")

if __name__ == "__main__":
    consolidate_embeddings()
