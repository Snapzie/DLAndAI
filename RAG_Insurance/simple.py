from sentence_transformers import SentenceTransformer
import numpy as np

# Load the pre-trained model for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # You can replace this with a domain-specific model if available

# Define the insurance policy chunks
insurance_policy_chunks = {
    1: "This policy provides coverage for damages to property resulting from natural disasters, including hurricanes, tornadoes, and earthquakes.",
    2: "The insured party is responsible for maintaining the property and performing necessary repairs in the event of a covered loss.",
    3: "Liability coverage is included, protecting the insured against claims for bodily injury or property damage caused to others.",
    4: "Exclusions: The policy does not cover damages caused by war, nuclear accidents, or intentional acts of vandalism by the insured.",
    5: "In the event of a claim, the policyholder must notify the insurance company within 30 days of the incident.",
    6: "The deductible amount for this policy is $500 for property damage and $1,000 for liability claims.",
    7: "This insurance policy provides coverage for both personal property and any permanently attached fixtures or structures on the premises.",
    8: "Policy limits are capped at $1 million for property damage and $500,000 for bodily injury claims.",
    9: "Optional coverage for flood damage can be added for an additional premium. This covers damages from heavy rainfall and river overflow.",
    10: "Claims will be processed according to the policy's terms, and the insurer reserves the right to investigate the cause of loss before approving any payments."
}

# Embed the context chunks
chunk_embeddings = {}
for key, value in insurance_policy_chunks.items():
    chunk_embeddings[key] = embedder.encode(value)  # Get the embedding for each chunk

# Convert the embeddings to a numpy array for easier manipulation
chunk_embeddings_array = np.array(list(chunk_embeddings.values()))

print("Embedded context chunks:", chunk_embeddings_array)

# Example user input
user_input = "What is covered under the liability section of the policy?"

# Embed the user input
user_input_embedding = embedder.encode(user_input)

print("User input embedding:", user_input_embedding)

from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarities between user input and each context chunk
cosine_similarities = cosine_similarity([user_input_embedding], chunk_embeddings_array)

# Get the index of the most relevant chunk
most_relevant_index = np.argmax(cosine_similarities)

# Extract the most relevant chunk
most_relevant_chunk_key = list(chunk_embeddings.keys())[most_relevant_index]
most_relevant_chunk = insurance_policy_chunks[most_relevant_chunk_key]

print(f"Most relevant chunk: {most_relevant_chunk_key}: {most_relevant_chunk}")