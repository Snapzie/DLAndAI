import chromadb
from sentence_transformers import SentenceTransformer
import json

def parse_json(filename):
    '''
    Prepares .json file for upload to chromadb by splitting the text, ids and section + page into three
    separate lists.
    \nInput:
    - filename str: Name of the file to read including path
    \nReturns:
    - texts str: a list of texts
    - metadata dict[str, str]: dictionary containing metadata
    - ids str: ids for each json element read
    '''
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    metadata = []
    ids = []
    for doc_id, content in data.items():
        texts.append(content["text"])
        metadata.append({
            "section": content.get("section", "Unknown"),
            "page": content.get("page", None),
            "name": content.get("name",-1)
        })
        ids.append(doc_id)
    return texts,metadata,ids

def init_nonpersistent_cosine_db():
    '''
    Initializes a non-persistent chromadb session with data from 'GPTPolicy.json' and 'RiskManagement.json'.
    \nThe session is initialized to use cosine-similarity and texts are explicitly embedded using 'all-MiniLM-L6-v2'.
    \nReturns:
    - chromadb collection: Database object
    '''
    for filename in ['GPTPolicy.json','RiskManagement.json']:
        texts,metadata,ids = parse_json(filename)

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