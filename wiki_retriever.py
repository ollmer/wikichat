import faiss
import json
import os
import numpy as np
import zipfile
from pathlib import Path

from sentence_transformers import SentenceTransformer


class Retriever:
    bge_prefix = "Represent this sentence for searching relevant passages: "
    def __init__(self, root):
        assert os.path.exists(root)
        root = Path(root)
        self.archive = zipfile.ZipFile(root / "data/en/paragraphs.zip", "r")
        self.index = faiss.read_index(str(root / "data/en/embs_IVF16384_HNSW32_2lvl_full.idx"))
        self.index.nprobe = 128
        self.model = SentenceTransformer("BAAI/bge-small-en", device="cuda")
        self.model.max_seq_length = 512

    def get_paragraph_by_vec_idx(self, vec_idx):
        chunk_id = vec_idx // 100000
        line_id = vec_idx % 100000
        with self.archive.open("enwiki_paragraphs_clean/enwiki_paragraphs_%03d.jsonl" % chunk_id) as f:
            for i, l in enumerate(f):
                if i == line_id:
                    paragraph = json.loads(l)
                    break
        return paragraph

    def search(self, query, k=5):
        emb = self.model.encode(self.bge_prefix + query, normalize_embeddings=True)
        _, neighbors = self.index.search(emb[np.newaxis, ...], k)
        results = []
        for n in neighbors[0]:
            paragraph = self.get_paragraph_by_vec_idx(n)
            results.append(paragraph)
        return results
