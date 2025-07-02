import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_context(user_input,db,query_document):
    '''
    Searches the provided chromadb for relevant context chunks.
    The strategy used involves finding the most similar match to the user input using cosine similarity. 
    Due to diminishing returns of multiplication with small numbers the 10 smallest distances provided
    by Chromadb are made into similarities (ranging 0-1). All similarities which are larger than the best
    similarity * 0.7 are kept as context.
    If the best similarity is less than 0.5, no context is returned.
    \nInput:
    - user_input str: Embedded prompt from the user
    - db chromadb instance: database object to be queried
    \nreturns:
    - str, str: the found context, the metadata of the found context
    '''
    results = db.query(
        query_embeddings=user_input,
        n_results=10,
        where={"name": query_document}
    )

    # chroma db returns distances as: 1-cosine_similarity. As distances are closer to zero, we experience diminishing returns
    # on comparisons. Thus, we convert the distances to similarities before comparison
    best = max(map(lambda x: 1-x,results['distances'][0]))
    if best < 0.5:
        return '', ''
    filtered = [
        {
            "id": results['ids'][0][i],
            "document": results['documents'][0][i],
            "distance": results['distances'][0][i],
            "metadata": results['metadatas'][0][i]
        }
        for i in range(len(results['ids'][0]))
        if 1-results['distances'][0][i] >= best * 0.7
    ]
    # print([1-e['distance'] for e in filtered]) # Debug
    context = '\n'.join([e['document'] for e in filtered])
    metadata = [e['metadata'] for e in filtered]
    return context,metadata