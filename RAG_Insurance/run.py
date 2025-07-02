from Model import InsuranceLLM, ModelConfig
from Database import init_nonpersistent_cosine_db
import Utility as util
from sentence_transformers import SentenceTransformer

# Global variables
config = ModelConfig()
llm = InsuranceLLM(config)
llm.load_model()
embedder = SentenceTransformer('all-MiniLM-L6-v2')
db = init_nonpersistent_cosine_db()

try:
    print("\nWelcome to Open-Insurance-LLM!")
    query_document = llm.console.input("[bold cyan]Document ('GPT': GPTPolicy, 'Risk': RiskManagement): [/bold cyan]").strip()
    while True:
        try:
            question = llm.console.input("[bold cyan]User: [/bold cyan]").strip()

            user_input_embedding = embedder.encode(question)
            context,metadata = util.get_context(user_input_embedding,db,query_document)
            # print(context) # Debug
            llm.generate_answer(question, context, metadata)
            print()  # Add a blank line after each response
            
        except Exception as e:
            llm.console.print(f"\n[red]Error processing input: {str(e)}[/red]")
            continue
except Exception as e:
    llm.console.print(f"\n[red]Fatal error: {str(e)}[/red]")
finally:
    if llm.llm_ctx:
        del llm.llm_ctx