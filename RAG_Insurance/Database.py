import chromadb
from sentence_transformers import SentenceTransformer
import json

def parse_json():
    with open("documents.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = []
    metadata = []
    ids = []

    for doc_id, content in data.items():
        texts.append(content["text"])
        metadata.append({
            "section": content.get("section", "Unknown"),
            "page": content.get("page", None)
        })
        ids.append(doc_id)
    return texts,metadata,ids

def init_nonpersistent_cosine_db():
    texts,metadata,ids = parse_json()

    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="chatgpt_paragraphs",
        configuration = {
            "hnsw": {
                "space": "cosine"
            }
        }
    )

    # texts = open('./Chunks.txt').read().split('\n')
    model = SentenceTransformer('all-MiniLM-L6-v2') # Also default for chromadb
    embeddings = [model.encode(t) for t in texts]

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadata,
        # ids=[str(i) for i in range(1,len(texts)+1)]
        ids=ids
    )
    return collection

# query = "What type of damages are covered under the property damage section of the policy?"
# query_embedding = model.encode(query)

# results = collection.query(
#     query_embeddings=[query_embedding],
#     n_results=10
# )

# print(results)