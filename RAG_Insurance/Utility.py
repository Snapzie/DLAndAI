import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_context(user_input_embedding,chunk_embeddings,insurance_policy_chunks):
    chunk_embeddings_array = np.array(list(chunk_embeddings.values()))
    cosine_similarities = cosine_similarity([user_input_embedding], chunk_embeddings_array)

    top_k_indices = np.argsort(cosine_similarities[0])[::-1][:10]
    top_similarities = cosine_similarities[0][top_k_indices]
    print(top_k_indices)
    print(top_similarities)
    # print(cosine_similarities[0])

    context_cand = []
    for i,idx in enumerate(top_k_indices):
        best = top_similarities[0]
        current = top_similarities[i]
        if current >= best * 0.7:
            context_cand.append(insurance_policy_chunks[list(chunk_embeddings.keys())[idx]])
    # top_k_chunks = {i: insurance_policy_chunks[list(chunk_embeddings.keys())[i]] for i in top_k_indices if abs(cosine_similarities[0][i]) >= 0.25}
    return '\n'.join(context_cand[::-1])