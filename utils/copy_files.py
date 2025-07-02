import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def copy_files(
    source_dir: str = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/DocumentRetrieval/data/pdf",
    target_dir: str = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/DocumentRetrieval/data/test_repo/pdf",
    max_files: int = 200,
    file_extensions: tuple = (".pdf", ".png")
) -> None:
    """
    Copy up to 200 files with specified extensions from source to target directory.

    Args:
        source_dir (str): Source directory containing files.
        target_dir (str): Target directory to copy files to.
        max_files (int): Maximum number of files to copy.
        file_extensions (tuple): File extensions to copy (e.g., '.jpg', '.png').
    """
    # Resolve paths
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Ensure source directory exists
    if not source_path.exists() or not source_path.is_dir():
        logger.error(f"Source directory {source_path} does not exist or is not a directory")
        return

    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)

    # Get list of files with specified extensions
    files = [
        f for f in source_path.iterdir()
        if f.is_file() and f.suffix.lower() in file_extensions
    ]

    if not files:
        logger.warning(f"No files with extensions {file_extensions} found in {source_path}")
        return

    logger.info(f"Found {len(files)} files in {source_path}")

    # Limit to max_files
    files_to_copy = files[:max_files]
    copied_count = 0

    # Copy files
    for file in files_to_copy:
        try:
            target_file = target_path / file.name
            shutil.copy2(file, target_file)  # copy2 preserves metadata
            logger.info(f"Copied {file.name} to {target_file}")
            copied_count += 1
        except Exception as e:
            logger.error(f"Failed to copy {file.name}: {str(e)}")
            continue

    logger.info(f"Completed copying {copied_count} files to {target_path}")

if __name__ == "__main__":
    copy_files()