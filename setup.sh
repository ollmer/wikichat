#!/bin/bash
pip install -r requirements.txt
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && LLAMA_METAL=1 make && cd ..
mkdir -p wiki_bge_small_en_embeddings/data/en/
echo "Downloading wiki index..."
wget https://huggingface.co/datasets/olmer/wiki_bge_small_en_embeddings/resolve/main/data/en/embs_IVF16384_HNSW32_2lvl_full.idx -P ./wiki_bge_small_en_embeddings/data/en/
wget https://huggingface.co/datasets/olmer/wiki_bge_small_en_embeddings/resolve/main/data/en/paragraphs.zip -P ./wiki_bge_small_en_embeddings/data/en/
mkdir -p models
echo "Downloading model..."
wget https://huggingface.co/TheBloke/Nous-Hermes-2-SOLAR-10.7B-GGUF/blob/main/nous-hermes-2-solar-10.7b.Q6_K.gguf -P ./models/

echo "Setup complete! Try to run 'python chat.py'"
