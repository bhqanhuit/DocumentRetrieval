import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from pdf2image import convert_from_path
from colpali_engine.models import ColQwen2, ColQwen2Processor
import os
from tqdm import tqdm
import json

model_name = "vidore/colqwen2-v1.0"
doc_dir = 'data/test_repo/pdf'
save_dir = 'data/preprocessed_documents/'
embedding_save_path = 'data/image_embeddings.pt'
metadata_save_path = 'data/image_metadata.json'
batch_size = 10

os.makedirs(save_dir, exist_ok=True)

# Step 1: Convert PDFs to images
documents = os.listdir(doc_dir)
for document in documents:
    doc_path = os.path.join(doc_dir, document)
    doc_images = convert_from_path(doc_path)
    for i, img in enumerate(doc_images):
        img.save(os.path.join(save_dir, f"{document[:-4]}_page{i}.jpg"), 'JPEG')

# Step 2: Load model and processor
model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()

processor = ColQwen2Processor.from_pretrained(model_name)

# Step 3: Process image files in batches
image_files = sorted([f for f in os.listdir(save_dir) if f.endswith('.jpg')])
all_image_embeddings = []
all_image_metadata = []

for i in tqdm(range(0, len(image_files), batch_size)):
    batch_files = image_files[i:i + batch_size]
    images = [Image.open(os.path.join(save_dir, fname)).convert("RGB") for fname in batch_files]
    
    batch_inputs = processor.process_images(images).to(model.device)

    with torch.no_grad():
        image_embeddings = model(**batch_inputs)

    all_image_embeddings.append(image_embeddings)
    all_image_metadata.extend(batch_files)  # Store filenames as metadata

# Combine and save image embeddings
image_embeddings = torch.cat(all_image_embeddings, dim=0)
torch.save(image_embeddings.cpu(), embedding_save_path)  # Save on CPU to avoid device issues

# Save metadata (file names)
with open(metadata_save_path, 'w') as f:
    json.dump(all_image_metadata, f)

# === You can load later like this ===
# image_embeddings = torch.load(embedding_save_path)
# with open(metadata_save_path) as f:
#     all_image_metadata = json.load(f)

# Step 4: Process queries
queries = [
    "i want to know about how attention work",
    "what is colipali"
]
query_inputs = processor.process_queries(queries).to(model.device)

with torch.no_grad():
    query_embeddings = model(**query_inputs)

# Step 5: Compute scores
scores = processor.score_multi_vector(query_embeddings, image_embeddings.to(model.device))

# Step 6: Show top result per query
topk = 3
for i, query in enumerate(queries):
    print(f"\nTop results for query: {query}")
    top_indices = torch.topk(scores[i], k=topk).indices
    for rank, idx in enumerate(top_indices):
        print(f"{rank + 1}. {all_image_metadata[idx]}")
