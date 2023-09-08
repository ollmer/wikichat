#!/bin/bash
pip install -r requirements.txt
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && LLAMA_METAL=1 make && cd ..
mkdir -p wiki_bge_small_en_embeddings/data/en/
echo "Downloading wiki index..."
wget https://huggingface.co/datasets/olmer/wiki_bge_small_en_embeddings/resolve/main/data/en/embs_IVF16384_HNSW32_2lvl_full.idx -P ./wiki_bge_small_en_embeddings/data/en/
wget https://huggingface.co/datasets/olmer/wiki_bge_small_en_embeddings/resolve/main/data/en/paragraphs.zip -P ./wiki_bge_small_en_embeddings/data/en/
mkdir -p models
echo "Downloading model..."
wget https://huggingface.co/TheBloke/StableBeluga-7B-GGUF/resolve/main/stablebeluga-7b.Q4_K_M.gguf -P ./models/

# if you have a 20+ GB or RAM and want a better model:
# wget https://huggingface.co/TheBloke/Stable-Platypus2-13B-GGUF/resolve/main/stable-platypus2-13b.Q4_K_M.gguf -P ./models/

echo "Setup complete! Try to run 'python chat.py'"
