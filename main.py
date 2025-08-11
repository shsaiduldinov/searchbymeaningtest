# from datasets import load_dataset
#
# # This will download the full Wikipedia dump (~9.57 GB) to your local cache
# dataset = load_dataset("izumi-lab/wikipedia-en-20230720", split="train")
#
# # Save as a local Parquet file
# dataset.to_parquet("wikipedia_full.parquet")
#
# print("Wikipedia dump saved as wikipedia_full.parquet")



import pandas as pd
import numpy as np
import math
from sentence_transformers import SentenceTransformer
from multiprocessing.pool import ThreadPool
from dotenv import load_dotenv
import os
import faiss


load_dotenv()


hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACE_TOKEN not found in .env file")

model = SentenceTransformer("all-MiniLM-L6-v2", use_auth_token=hf_token)

def encode_chunk(chunk):
    return model.encode(chunk, batch_size=256, show_progress_bar=True)

def search(query, index, texts, top_k=5):
    query_embedding = model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]), top_k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append((texts[idx], dist))
    return results

if __name__ == '__main__':
    df = pd.read_parquet("sample_wiki.parquet")
    print(df.head())

    num_threads = 2
    chunk_size = math.ceil(len(df) / num_threads)
    texts = df['text'].tolist()
    text_chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    with ThreadPool(num_threads) as pool:
        chunk_embeddings = pool.map(encode_chunk, text_chunks)
    embeddings = np.vstack(chunk_embeddings).astype("float32")

    np.save("embeddings.npy", embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    search_query = input("Enter search query: ")
    for text, score in search(search_query, index, texts):
        print(f"Score: {score:.4f} | Text: {text[:100]}...")
