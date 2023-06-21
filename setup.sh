#!/bin/bash
pip install -r requirements.txt
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.65 --force-reinstall --upgrade --no-cache-dir
mkdir -p wiki_mpnet_index/data/en/
echo "Downloading wiki index..."
wget https://huggingface.co/datasets/olmer/wiki_mpnet_index/resolve/main/data/en/embs_IVF16384_HNSW32_2lvl_full.idx -P ./wiki_mpnet_index/data/en/
wget https://huggingface.co/datasets/olmer/wiki_mpnet_index/resolve/main/data/en/paragraphs.zip -P ./wiki_mpnet_index/data/en/
mkdir models
echo "Downloading model..."
wget https://huggingface.co/TheBloke/Wizard-Vicuna-13B-Uncensored-GGML/resolve/main/Wizard-Vicuna-13B-Uncensored.ggmlv3.q6_K.bin -P ./models/
echo "Setup complete!"