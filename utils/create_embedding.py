import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from pdf2image import convert_from_path
from colpali_engine.models import ColQwen2, ColQwen2Processor
from pathlib import Path
import json
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import logging
import multiprocessing
from typing import List, Dict, Any
from functools import partial

# Configure process-safe logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(processName)s (PID:%(process)d) - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
project_root = Path(__file__).parent.parent

def process_pdf(document: Path, save_path: Path, resize_dims: tuple = (224, 224)) -> List[Dict[str, Any]]:
    """
    Process a single PDF file: convert to images, resize them, and generate metadata.
    
    Args:
        document (Path): Path to the PDF file.
        save_path (Path): Directory to save temporary images.
        resize_dims (tuple): Dimensions to resize images to (width, height).
    
    Returns:
        List[Dict[str, Any]]: List of metadata dictionaries for each page.
    """
    metadata = []
    try:
        logger.info(f"Converting {document.name}")
        doc_images = convert_from_path(str(document))
        for i, img in enumerate(doc_images):
            img_path = save_path / f"{document.stem}_page{i}.jpg"
            # Resize image to consistent dimensions
            img = img.resize(resize_dims, Image.Resampling.LANCZOS)
            img.save(img_path, "JPEG")
            metadata.append({
                "file_path": str(document),
                "page_number": i + 1,
                "image_file": str(img_path.name),
                "embedding_file": f"{document.stem}_page{i}_embedding.pt"
            })
    except Exception as e:
        logger.error(f"Failed to convert {document.name}: {str(e)}")
    return metadata

def preprocess_images(
    input_dir: str = "data/pdf",
    save_dir: str = "data/preprocessed_documents",
    metadata_save_path: str = "data/doc_embedding/image_metadata.json",
    num_processes: int = None
) -> List[Dict[str, Any]]:
    """
    Preprocess PDFs by converting them to images and saving metadata.
    
    Args:
        input_dir (str): Directory containing PDF files.
        save_dir (str): Directory to save temporary images.
        metadata_save_path (str): Path to save metadata (.json file).
        num_processes (int, optional): Number of processes for parallel PDF conversion.
    
    Returns:
        List[Dict[str, Any]]: Metadata for processed images.
    """
    input_path = Path(input_dir)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting PDFs in {input_dir} to images")
    documents = [f for f in input_path.glob("*.pdf")]
    if not documents:
        logger.warning(f"No PDF files found in {input_dir}")
        return []

    num_processes = num_processes or multiprocessing.cpu_count()
    logger.info(f"Using {num_processes} processes for PDF conversion")

    process_pdf_partial = partial(process_pdf, save_path=save_path, resize_dims=(224, 224))
    metadata = []

    for document in tqdm(documents, desc="Converting PDFs to images", total=len(documents)):
        result = process_pdf(document, save_path, resize_dims=(224, 224))
        metadata.extend(result)

    if not metadata:
        logger.warning("No images generated")
        return []

    logger.info(f"Generated {len(metadata)} images from {len(documents)} PDFs")

    # Save metadata
    logger.info(f"Saving metadata to {metadata_save_path}")
    with open(metadata_save_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata

def generate_embedding_for_image(
    image_file: str,
    embedding_file: str,
    save_dir: str,
    model: ColQwen2,
    processor: ColQwen2Processor,
    device: torch.device,
    dummy_shape: tuple = (1, 75, 128)  # Updated based on provided embedding shape
) -> bool:
    """
    Generate embedding for a single image and save it to a file.
    If processing fails, save a dummy embedding.

    Args:
        image_file (str): Name of the image file.
        embedding_file (str): Name of the output embedding file.
        save_dir (str): Directory containing images and to save embeddings.
        model (ColQwen2): Pretrained model for embedding generation.
        processor (ColQwen2Processor): Processor for image preprocessing.
        device (torch.device): Device to run the model on.
        dummy_shape (tuple): Shape of the dummy embedding (batch_size, height, width).

    Returns:
        bool: True if successful (real or dummy embedding saved), False otherwise.
    """
    save_path = Path(save_dir)
    embedding_path = save_path / embedding_file

    try:
        # Load and preprocess image
        img = Image.open(save_path / image_file).convert("RGB")
        batch_inputs = processor.process_images([img]).to(device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = model(**batch_inputs)
        
        # Save embedding
        print(f"{embedding.shape} is the shape of embedding")
        torch.save(embedding.cpu(), embedding_path)
        logger.info(f"Saved embedding for {image_file} to {embedding_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to process {image_file}: {str(e)}")
        # Create and save dummy embedding
        try:
            dummy_embedding = torch.zeros(dummy_shape)
            torch.save(dummy_embedding, embedding_path)
            logger.info(f"Saved dummy embedding for {image_file} to {embedding_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save dummy embedding for {image_file}: {str(e)}")
            return False

def generate_embeddings(
    save_dir: str = "data/preprocessed_documents",
    metadata_save_path: str = "data/doc_embedding/image_metadata.json",
    model_name: str = "vidore/colqwen2-v1.0"
) -> None:
    """
    Generate embeddings for preprocessed images, saving each as a separate file.
    If an image fails, save a dummy embedding.

    Args:
        save_dir (str): Directory containing preprocessed images and to save embeddings.
        metadata_save_path (str): Path to load metadata (.json file).
        model_name (str): Name of the pretrained ColQwen2 model.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Load metadata
    try:
        with open(metadata_save_path, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metadata from {metadata_save_path}: {str(e)}")
        return

    if not metadata:
        logger.warning("No metadata found")
        return

    # Load model and processor
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

    # Process each image individually
    logger.info(f"Processing {len(metadata)} images")
    success_count = 0
    for item in tqdm(metadata, desc="Generating embeddings"):
        success = generate_embedding_for_image(
            image_file=item["image_file"],
            embedding_file=item["embedding_file"],
            save_dir=save_dir,
            model=model,
            processor=processor,
            device=device,
            dummy_shape=(1, 75, 128)  # Updated based on provided embedding shape
        )
        if success:
            success_count += 1

    logger.info(f"Successfully processed {success_count}/{len(metadata)} images")

if __name__ == "__main__":
    # Step 1: Preprocess images and save metadata
    metadata = preprocess_images(num_processes=8)
    
    # Step 2: Generate embeddings for preprocessed images
    if metadata:
        generate_embeddings()