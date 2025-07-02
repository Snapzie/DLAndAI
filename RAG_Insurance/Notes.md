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
- Which embedding to use?
- Which similarity strategy to use?
- How to summarize doc
- Chunk sizes?
- Page breaks
- Markdown tables in PDF
- User interface

### Questions
- Explain everything the insurer should do **
- What does the product development process generally involve?
- Which measures should an insurer put in place to identify risks?
- Give examples of underwriting guidelines
- Who should establish the risk management framework?
- Explain the purpose of risk policies

- Does the policy cover liability for bodily injuries caused to others on the insured property?
- What natural disasters are covered under the natural disaster section of the policy?
- Does the policy cover medical expenses for visitors injured on the insured property?
- What damages are covered by the earthquake coverage in the policy?
- Are injuries caused by the insured's intentional actions covered under the liability section of the policy? -- Shows llm reasoning
- What types of theft or vandalism are not covered under the policy? -- Good answer, but then the added exclusion post-addition seem to screw it
- Are damages caused by arson covered under the fire damage section? -- Confused about arson is described in the policy but not covered