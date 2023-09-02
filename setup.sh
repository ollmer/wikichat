#!/bin/bash
pip install -r requirements.txt
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.83 --force-reinstall --upgrade --no-cache-dir
mkdir -p wiki_bge_small_en_embeddings/data/en/
echo "Downloading wiki index..."
wget https://huggingface.co/datasets/olmer/wiki_bge_small_en_embeddings/resolve/main/data/en/embs_IVF16384_HNSW32_2lvl_full.idx -P ./wiki_bge_small_en_embeddings/data/en/
wget https://huggingface.co/datasets/olmer/wiki_bge_small_en_embeddings/resolve/main/data/en/paragraphs.zip -P ./wiki_bge_small_en_embeddings/data/en/
mkdir models
echo "Downloading model..."
wget https://huggingface.co/TheBloke/Stable-Platypus2-13B-GGML/resolve/main/stable-platypus2-13b.ggmlv3.q4_K_M.bin -P ./models/
echo "Setup complete!"