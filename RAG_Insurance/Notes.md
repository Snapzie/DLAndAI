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
- Convert all parsed data to lower-case in case model cites context?
- When should the model answer, idk?

### Questions
- Explain everything the insurer should do **
- What does the product development process generally involve?
- Which measures should an insurer put in place to identify risks?
- Give examples of underwriting guidelines
- Who should establish the risk management framework?
- Explain the purpose of risk policies