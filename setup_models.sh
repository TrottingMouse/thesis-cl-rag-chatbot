#!/bin/bash
# setup_models.sh

echo "Creating model directory..."
mkdir -p storage/models

echo "Downloading Jina Embeddings v5 Nano..."
git clone https://huggingface.co/jinaai/jina-embeddings-v5-text-nano storage/models/jina-embeddings

echo "Applying exact config_class patch to the architecture file..."
FILE="storage/models/jina-embeddings/modeling_jina_embeddings_v5.py"

# Inject the specific Jina configuration import and class assignment
sed -i 's/class JinaEmbeddingsV5Model(PeftMixedModel):/from .configuration_jina_embeddings_v5 import JinaEmbeddingsV5Config\n\nclass JinaEmbeddingsV5Model(PeftMixedModel):\n    config_class = JinaEmbeddingsV5Config/' $FILE

echo "Purging Hugging Face sandbox cache..."
rm -rf ~/.cache/huggingface/modules/transformers_modules/jina_hyphen_embeddings

echo "Setup complete! The model is ready for the offline pipeline."