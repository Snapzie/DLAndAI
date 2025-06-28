from Model import InsuranceLLM, ModelConfig
import Utility as util
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

config = ModelConfig()
llm = InsuranceLLM(config)
llm.load_model()
embedder = SentenceTransformer('all-MiniLM-L6-v2')

insurance_policy_chunks = {k:v for k,v in enumerate(open('./Chunks.txt').read().split('\n'))}
chunk_embeddings = {k:embedder.encode(v) for k,v in insurance_policy_chunks.items()}

questions = [
    "What type of damages are covered under the property damage section of the policy?",
    "Does the policy cover liability for bodily injuries caused to others on the insured property?",
    "What does the flood damage coverage protect against?",
    "How does the policy address damages caused by theft or vandalism?",
    "What types of fire damage are covered under the policy?",
    "How does the policy compensate for lost rental income if the property becomes uninhabitable?",
    "What natural disasters are covered under the natural disaster section of the policy?",
    "Does the policy cover medical expenses for visitors injured on the insured property?",
    "What is covered under the legal defense coverage section of the policy?",
    "What damages are covered by the earthquake coverage in the policy?",
    "What damages are excluded from the property damage coverage?",
    "Are injuries caused by the insured's intentional actions covered under the liability section of the policy?",
    "Does the policy cover damages caused by sewer backups during a flood?",
    "What types of theft or vandalism are not covered under the policy?",
    "Are damages caused by arson covered under the fire damage section?",
    "What exclusions are associated with the loss of rent coverage?",
    "Is damage resulting from a failure to follow local safety ordinances covered in the natural disaster section?",
    "Are injuries sustained while engaging in illegal activities on the insured property covered under the medical payments section?",
    "What lawsuits are excluded from the legal defense coverage under the policy?",
    "What types of damages are excluded under the earthquake coverage section?"
]

question_avg = []
context_avg = []
for q in questions:
    llm.console.print(f"[bold cyan]User:[/bold cyan] {q}")
    q_embed = embedder.encode(q)
    context = util.get_context(q_embed,chunk_embeddings,insurance_policy_chunks)
    response = llm.generate_answer(q, context)

    question_avg.append(float(abs(cosine_similarity([q_embed],[embedder.encode(response)])[0][0])))
    context_avg.append(float(abs(cosine_similarity([embedder.encode(context)],[embedder.encode(response)])[0][0])))
llm.console.print(f'[bold green]Question avg. similarity:[/bold green][default] {np.mean(question_avg):.2f}\n{list(map(lambda x: f"{x:.2f}",question_avg))}[/default]')
llm.console.print(f'[bold green]Context avg. similarity:[/bold green][default] {np.mean(context_avg):.2f}\n{list(map(lambda x: f"{x:.2f}",context_avg))}[/default]')

if llm.llm_ctx:
    del llm.llm_ctx