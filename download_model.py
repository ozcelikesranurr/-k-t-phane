import os
print("=== Start Downloading SentenceTransformer Model ===")
from sentence_transformers import SentenceTransformer
# This will download the model to the default cache folder during build time
model = SentenceTransformer('all-MiniLM-L6-v2')
print("=== Model Download Complete and Cached successfully ===")
