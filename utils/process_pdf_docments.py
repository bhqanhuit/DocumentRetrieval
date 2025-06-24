from pdf2image import convert_from_path
from pathlib import Path
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def preprocess_pdfs(
    input_dir: str = "data/test_repo/pdf",
    save_dir: str = "data/preprocessed_documents"
) -> None:
    """
    Convert PDF files to JPEG images and save them to the specified directory.
    
    Args:
        input_dir (str): Directory containing PDF files (relative to project root).
        save_dir (str): Directory to save JPEG images (relative to project root).
    """
    # Get project root directory (parent of utils/)
    project_root = Path(__file__).parent.parent

    # Resolve paths relative to project root
    input_path = project_root / input_dir
    save_path = project_root / save_dir

    # Ensure save directory exists
    save_path.mkdir(parents=True, exist_ok=True)

    # Get list of PDF files
    documents = [f for f in input_path.glob("*.pdf")]
    if not documents:
        logger.warning(f"No PDF files found in {input_path}")
        return

    logger.info(f"Found {len(documents)} PDF files in {input_path}")

    # Process each PDF
    for document in documents:
        doc_path = document
        logger.info(f"Converting {document.name}")
        try:
            # Convert PDF to images
            doc_images = convert_from_path(str(doc_path))
            # Save each page as JPEG
            for i, img in enumerate(doc_images):
                img_path = save_path / f"{document.stem}_page{i}.jpg"
                img.save(img_path, "JPEG")
                logger.info(f"Saved {img_path.name}")
        except Exception as e:
            logger.error(f"Failed to convert {document.name}: {str(e)}")
            continue

    logger.info(f"Completed preprocessing. Images saved to {save_path}")

if __name__ == "__main__":
    preprocess_pdfs()