import torch
from embed_anything import EmbedData, ColpaliModel
import numpy as np
from tabulate import tabulate
from pathlib import Path

model: ColpaliModel = ColpaliModel.from_pretrained_onnx("akshayballal/colpali-v1.2-merged-onnx")


directory = Path("data/test_repo/Pdf")
files = list(directory.glob("*.pdf"))
file_embed_data: list[EmbedData] = []
for idx, file in enumerate(files):
      if (idx > 100): break
      print(f"Embedding file: {file}")
      try:
        embedding: list[EmbedData] = model.embed_file(str(file), batch_size=8)
        file_embed_data.extend(embedding)
      except:
          print("can not extract this")

query = "some thing related to medical"

query_embedding = model.embed_query(query)
query_embeddings = np.array([e.embedding for e in query_embedding])

file_embeddings = np.array([e.embedding for e in file_embed_data])
scores = np.einsum("bnd,csd->bcns", query_embeddings, file_embeddings).max(axis=3).sum(axis=2).squeeze()

top_pages = np.argsort(scores)[::-1][:5]

table = [
    [file_embed_data[page].metadata["file_path"], file_embed_data[page].metadata["page_number"]]
    for page in top_pages
]

print(tabulate(table, headers=["File Name", "Page Number"], tablefmt="grid"))