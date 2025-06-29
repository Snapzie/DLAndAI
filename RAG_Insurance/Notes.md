### How to ensure exclusions are mentioned?
- Always add "Any exclusions or exceptions?" to the prompt

### Improvements
- Virtual environment with requirements.txt
- Prepend all answers with: "The information for this answer was found on the following sections and pages:..."

### TODO
- Meta data
- Multiple documents
- Parser
- Chunking
- WebServer

### Chromadb
- client = chromadb.PersistentClient(path="./chroma_db")
- client.delete_collection(name="test")
- query_embeddings=[query_embedding] or query_texts = [text]


### Considerations
- Should metadata be part of complete_response?
    - Should it be part of model history?

0.7984
0.8396

0.7905
0.7256

[0.7507878541946411, 0.7245298027992249, 0.7083057761192322]