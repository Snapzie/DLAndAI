from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import numpy as np

documents = [
    {"content": "Genetics is the study of genes, genetic variation, and heredity in organisms. It is an important branch in biology because heredity is vital to organisms' evolution. Gregor Mendel, a Moravian Augustinian friar working in the 19th century in Brno, was the first to study genetics scientifically. Mendel studied \"trait inheritance\", patterns in the way traits are handed down from parents to offspring over time. He observed that organisms (pea plants) inherit traits by way of discrete \"units of inheritance\". This term, still used today, is a somewhat ambiguous definition of what is referred to as a gene.", "page_id": "Header", "URL":"https://en.wikipedia.org/wiki/Genetics"},
    {"content": "Prior to Mendel, Imre Festetics, a Hungarian noble, who lived in Kőszeg before Mendel, was the first who used the word \"genetic\" in hereditarian context, and is considered the first geneticist. He described several rules of biological inheritance in his work The genetic laws of nature (Die genetischen Gesetze der Natur, 1819). His second law is the same as that which Mendel published. In his third law, he developed the basic principles of mutation (he can be considered a forerunner of Hugo de Vries). Festetics argued that changes observed in the generation of farm animals, plants, and humans are the result of scientific laws. Festetics empirically deduced that organisms inherit their characteristics, not acquire them. He recognized recessive traits and inherent variation by postulating that traits of past generations could reappear later, and organisms could produce progeny with different attributes. These observations represent an important prelude to Mendel's theory of particulate inheritance insofar as it features a transition of heredity from its status as myth to that of a scientific discipline, by providing a fundamental theoretical basis for genetics in the twentieth century.", "page_id": "History", "URL":"https://en.wikipedia.org/wiki/Genetics"},
    {"content": "Although geneticists originally studied inheritance in a wide variety of organisms, the range of species studied has narrowed. One reason is that when significant research already exists for a given organism, new researchers are more likely to choose it for further study, and so eventually a few model organisms became the basis for most genetics research. Common research topics in model organism genetics include the study of gene regulation and the involvement of genes in development and cancer. Organisms were chosen, in part, for convenience—short generation times and easy genetic manipulation made some organisms popular genetics research tools. Widely used model organisms include the gut bacterium Escherichia coli, the plant Arabidopsis thaliana, baker's yeast (Saccharomyces cerevisiae), the nematode Caenorhabditis elegans, the common fruit fly (Drosophila melanogaster), the zebrafish (Danio rerio), and the common house mouse (Mus musculus).", "page_id": "Model organisms", "URL":"https://en.wikipedia.org/wiki/Genetics"}
]
retriever = SentenceTransformer('all-MiniLM-L6-v2')

doc_embeddings = retriever.encode([doc["content"] for doc in documents], convert_to_tensor=True)

model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Query:")
query = input()
query_embedding = retriever.encode(query, convert_to_tensor=True)

similarities = util.pytorch_cos_sim(query_embedding, doc_embeddings)
best_match_idx = np.argmax(similarities.cpu().numpy())
retrieved_doc = documents[best_match_idx]

prompt = f"Context: {retrieved_doc['content']}\n\nQuestion: {query}\nAnswer:"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

outputs = generator.generate(input_ids, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

answer_with_reference = {
    "answer": response,
    "reference": retrieved_doc["page_id"],
    "URL": retrieved_doc["URL"]
}

print(answer_with_reference)