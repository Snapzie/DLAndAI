import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# def get_context_nondb(user_input_embedding,chunk_embeddings,insurance_policy_chunks):
#     chunk_embeddings_array = np.array(list(chunk_embeddings.values()))
#     cosine_similarities = cosine_similarity([user_input_embedding], chunk_embeddings_array)

#     top_k_indices = np.argsort(cosine_similarities[0])[::-1][:10]
#     top_similarities = cosine_similarities[0][top_k_indices]
#     print(top_k_indices)
#     print(top_similarities)
#     # print(cosine_similarities[0])

#     context_cand = []
#     for i,idx in enumerate(top_k_indices):
#         best = top_similarities[0]
#         current = top_similarities[i]
#         if current >= best * 0.7:
#             context_cand.append(insurance_policy_chunks[list(chunk_embeddings.keys())[idx]])
#     # top_k_chunks = {i: insurance_policy_chunks[list(chunk_embeddings.keys())[i]] for i in top_k_indices if abs(cosine_similarities[0][i]) >= 0.25}
#     return '\n'.join(context_cand[::-1])

def get_context(user_input,db):
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
        n_results=10
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
    print([1-e['distance'] for e in filtered])
    context = '\n'.join([e['document'] for e in filtered])
    metadata = [e['metadata'] for e in filtered]
    return context,metadata