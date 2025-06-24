# Document Retrieval Project

This project implements a document retrieval system that processes PDF files, converts them to images, generates embeddings using the `ColQwen2` model, and provides a search API with a web interface. The system allows users to search for relevant PDF pages based on text queries, with results displayed in a user-friendly HTML table.

## Project Structure
project/\
├── app/\
│ ├── static/\
│ │ └── index.html # Web interface for search\
│ ├── main.py # FastAPI application\
│ ├── models.py # Pydantic models for API\
│ └── services.py # Document retrieval logic\
├── examples/\
│ ├── colpali_inference.py # Example inference script for ColPali model\
│ └── inference.py # General inference example script\
├── utils/\
│ ├── create_embedding.py # Script to generate embeddings\
│ └── preprocess_pdf_documents.py # Script to convert PDFs to images\
├── assets/\
│ ├── pdf/\
│ ├── preprocessed_documents/\
│ └── doc_embeddings/\
└── requirements.txt # Project dependencies\

## Features

- **PDF Preprocessing**: Converts PDFs to JPEG images for embedding.
- **Embedding Generation**: Uses the `ColQwen2` model to generate embeddings for PDF pages.
- **Search API**: FastAPI endpoint (`/search`) to query embeddings and return top matching PDF pages.
- **Web Interface**: HTML page with a search bar and results table, styled with Tailwind CSS.

## Prerequisites

- **Python**: 3.8 or higher
- **CUDA** (optional): For GPU acceleration with `ColQwen2` (falls back to CPU if unavailable)

## Installation

1. **Clone the Repository** (if applicable):

   ```bash
   git clone https://github.com/bhqanhuit/DocumentRetrieval.git
   cd document-retrieval
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes:

   ```
   fastapi==0.115.2
   uvicorn==0.32.0
   torch
   colpali_engine
   transformers
   pdf2image
   pillow
   tqdm
   python-multipart
   ```

3. **Prepare Input Directory**:

   - Place PDF files in `assets/pdf`.

## Usage

### 1. Preprocess PDFs

Convert PDF files to JPEG images:

```bash
python utils/preprocess_pdfs.py
```

- **Input**: PDFs in `data/pdf`.
- **Output**: JPEG images in `assets/preprocessed_documents` (e.g., `doc1_page0.jpg`).

### 2. Generate Embeddings

Generate embeddings for the images using the `ColQwen2` model:

```bash
python utils/create_embedding.py
```

- **Input**: JPEG images in `assets/preprocessed_documents`.
- **Output**:
  - Embeddings: `assets/doc_embeddings/image_embeddings.pt`
  - Metadata: `assets/doc_embeddings/image_metadata.json`

### 3. Run the FastAPI Application

Start the FastAPI server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- Access the web interface at `http://localhost:8000`.

### 4. Search Documents

- **Web Interface**:
  - Open `http://localhost:8000` in a browser.
  - Enter a query (e.g., "how attention works") in the search bar.
  - Click "Search" to view the top 5 matching PDF pages in a table.
- **API Endpoint**:
  - Endpoint: `POST /search`
  - Request: JSON body with a `query` field.
  - Example:

    ```bash
    curl -X POST "http://localhost:8000/search" -H "Content-Type: application/json" -d '{"query": "what is colipali"}'
    ```
  - Response: JSON array of objects with `file_path` and `page_number`.

## Example Workflow

1. Place PDFs in `data/pdf`.
2. Run `python utils/create_embedding.py` to generate embeddings.
3. Start the server with `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
4. Open `http://localhost:8000` and search for "what is attention?".

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Built with FastAPI, ColQwen2, and pdf2image.