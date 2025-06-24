import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from pdf2image import convert_from_path
from colpali_engine.models import ColQwen2, ColQwen2Processor
import os
from pathlib import Path
import json
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
project_root = Path(__file__).parent.parent


def create_embeddings(
    input_dir: str = "data/test_repo/pdf",
    save_dir: str = "data/preprocessed_documents",
    embedding_save_path: str = "data/doc_embedding/image_embeddings.pt",
    metadata_save_path: str = "data/doc_embedding/image_metadata.json",
    batch_size: int = 10,
    model_name: str = "vidore/colqwen2-v1.0"
) -> None:
    """
    Generate embeddings for PDF files by converting them to images and processing with ColQwen2.
    
    Args:
        input_dir (str): Directory containing PDF files.
        save_dir (str): Directory to save temporary images and outputs.
        embedding_save_path (str): Path to save embeddings (.pt file).
        metadata_save_path (str): Path to save metadata (.json file).
        batch_size (int): Batch size for processing images.
        model_name (str): Name of the pretrained ColQwen2 model.
    """
    # Initialize paths
    input_path = Path(input_dir)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert PDFs to images
    logger.info(f"Converting PDFs in {input_dir} to images")
    documents = [f for f in input_path.glob("*.pdf")]
    if not documents:
        logger.warning(f"No PDF files found in {input_dir}")
        return

    metadata = []
    for document in documents:
        logger.info(f"Converting {document.name}")
        try:
            doc_images = convert_from_path(str(document))
            for i, img in enumerate(doc_images):
                img_path = save_path / f"{document.stem}_page{i}.jpg"
                img.save(img_path, "JPEG")
                metadata.append({
                    "file_path": str(document),
                    "page_number": i + 1,
                    "image_file": str(img_path.name)
                })
        except Exception as e:
            logger.error(f"Failed to convert {document.name}: {str(e)}")
            continue

    if not metadata:
        logger.warning("No images generated")
        return

    # Step 2: Load model and processor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model {model_name} on {device}")
    try:
        model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
        processor = ColQwen2Processor.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Failed to load model or processor: {str(e)}")
        raise

    # Step 3: Process images in batches
    image_files = [m["image_file"] for m in metadata]
    all_image_embeddings = []

    logger.info(f"Processing {len(image_files)} images in batches of {batch_size}")
    for i in tqdm(range(0, len(image_files), batch_size)):
        batch_files = image_files[i:i + batch_size]
        images = [Image.open(save_path / fname).convert("RGB") for fname in batch_files]
        
        batch_inputs = processor.process_images(images).to(device)
        with torch.no_grad():
            image_embeddings = model(**batch_inputs)
        
        all_image_embeddings.append(image_embeddings.cpu())  # Move to CPU to save memory

    # Combine and save embeddings
    logger.info(f"Saving embeddings to {embedding_save_path}")
    image_embeddings = torch.cat(all_image_embeddings, dim=0)
    torch.save(image_embeddings, embedding_save_path)

    # Save metadata
    logger.info(f"Saving metadata to {metadata_save_path}")
    with open(metadata_save_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Processed {len(image_files)} pages from {len(documents)} PDFs")

if __name__ == "__main__":
    create_embeddings()