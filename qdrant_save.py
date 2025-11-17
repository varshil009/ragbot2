from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pickle
import os

host = os.getenv("QDRANT_HOST")
port = os.getenv("QDRANT_PORT")


client = QdrantClient(host=host, port=port)

def ensure_collection_exists(collection_name, vector_size):
    try:
        client.get_collection(collection_name)
    except Exception as e:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

def prepare_points(embeddings, texts):
    return [
        PointStruct(
            id=i,
            vector=embedding,
            payload={"text": text}
        )
        for i, (embedding, text) in enumerate(zip(embeddings, texts))
    ]

# Load DL data
with open(r"src/data_dl.pkl", "rb") as f:
    dl_data = pickle.load(f)
    dl_embeddings = dl_data['embeddings']
    dl_texts = dl_data['chunks']

# Ensure collection exists for DL data
ensure_collection_exists("data_dl_collection", dl_embeddings[0].shape[0])

# Upsert DL data
client.upsert(
    collection_name="data_dl_collection",
    points=prepare_points(dl_embeddings, dl_texts)
)
print("Completed saving DL data")

# Load DS data
with open(r"src/data_ds.pkl", "rb") as f:
    ds_data = pickle.load(f)
    ds_embeddings = ds_data['embeddings']
    ds_texts = ds_data['chunks']

# Ensure collection exists for DS data
ensure_collection_exists("data_ds_collection", ds_embeddings[0].shape[0])

# Upsert DS data
client.upsert(
    collection_name="data_ds_collection",
    points=prepare_points(ds_embeddings, ds_texts)
)
print("Completed saving DS data")