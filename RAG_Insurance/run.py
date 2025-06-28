from Model import InsuranceLLM, ModelConfig
import Utility as util
from sentence_transformers import SentenceTransformer

config = ModelConfig()
llm = InsuranceLLM(config)
llm.load_model()
embedder = SentenceTransformer('all-MiniLM-L6-v2')

insurance_policy_chunks = {k:v for k,v in enumerate(open('./Chunks.txt').read().split('\n'))}
chunk_embeddings = {k:embedder.encode(v) for k,v in insurance_policy_chunks.items()}

try:
    print("\nWelcome to Open-Insurance-LLM!")
    
    while True:
        try:
            question = llm.console.input("[bold cyan]User: [/bold cyan]").strip()

            user_input_embedding = embedder.encode(question)
            context = util.get_context(user_input_embedding,chunk_embeddings,insurance_policy_chunks)
            print(context)
            llm.generate_answer(question, context)
            print()  # Add a blank line after each response
            
        except Exception as e:
            llm.console.print(f"\n[red]Error processing input: {str(e)}[/red]")
            continue
except Exception as e:
    llm.console.print(f"\n[red]Fatal error: {str(e)}[/red]")
finally:
    if llm.llm_ctx:
        del llm.llm_ctx