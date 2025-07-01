import chromadb
from sentence_transformers import SentenceTransformer
import json

def parse_json():
    '''
    Prepares /output.json for upload to chromadb by splitting the text, ids and section + page into three
    separate lists.
    '''
    with open("output.json", "r", encoding="utf-8") as f:
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
    '''
    Initializes a non-persistent chromadb session with data from '/output.json'.
    The session is initialized to use cosine-similarity and texts are explicitly
    embedded using 'all-MiniLM-L6-v2'.
    '''
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

    model = SentenceTransformer('all-MiniLM-L6-v2') # Also default for chromadb
    embeddings = [model.encode(t) for t in texts]

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadata,
        ids=ids
    )
    return collection