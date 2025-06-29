# Attached under `Files and Versions` (inference_open-insurance-llm-gguf.py)
import os
import time
from pathlib import Path
from llama_cpp import Llama
from rich.console import Console
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

@dataclass
class ModelConfig:
    # Optimized parameters for coherent responses and efficient performance on devices like MacBook Air M2
    model_name: str = "Raj-Maharajwala/Open-Insurance-LLM-Llama3-8B-GGUF"
    model_file: str = "open-insurance-llm-q4_k_m.gguf"
    # model_file: str = "open-insurance-llm-q8_0.gguf"  # 8-bit quantization; higher precision, better quality, increased resource usage
    # model_file: str = "open-insurance-llm-q5_k_m.gguf"  # 5-bit quantization; balance between performance and resource efficiency
    max_tokens: int = 1000  # Maximum number of tokens to generate in a single output
    temperature: float = 0.1  # Controls randomness in output; lower values produce more coherent responses (performs scaling distribution)
    top_k: int = 15  # After temperature scaling, Consider the top 15 most probable tokens during sampling
    top_p: float = 0.2  # After reducing the set to 15 tokens, Uses nucleus sampling to select tokens with a cumulative probability of 20%
    repeat_penalty: float = 1.2  # Penalize repeated tokens to reduce redundancy
    num_beams: int = 4  # Number of beams for beam search; higher values improve quality at the cost of speed
    n_gpu_layers: int = -2  # Number of layers to offload to GPU; -1 for full GPU utilization, -2 for automatic configuration
    n_ctx: int = 2048  # Context window size; Llama 3 models support up to 8192 tokens context length
    n_batch: int = 256  # Number of tokens to process simultaneously; adjust based on available hardware (suggested 512)
    verbose: bool = False  # True for enabling verbose logging for debugging purposes
    use_mmap: bool = False  # Memory-map model to reduce RAM usage; set to True if running on limited memory systems
    use_mlock: bool = True  # Lock model into RAM to prevent swapping; improves performance on systems with sufficient RAM
    offload_kqv: bool = True  # Offload key, query, value matrices to GPU to accelerate inference



class InsuranceLLM:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.llm_ctx = None
        self.console = Console()
        self.conversation_history: List[Dict[str, str]] = []
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.system_message = (
            "This is a chat between a user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. "
            "The assistant should also indicate when the answer cannot be found in the context. "
            "You are an expert from the Insurance domain with extensive insurance knowledge and "
            "professional writer skills, especially about insurance policies. "
            "Your name is OpenInsuranceLLM, and you were developed by Raj Maharajwala. "
            "You are willing to help answer the user's query with a detailed explanation. "
            # "In your explanation, leverage your deep insurance expertise, such as relevant insurance policies, "
            # "complex coverage plans, or other pertinent insurance concepts. Use precise insurance terminology while "
            "Use precise insurance terminology while "
            "still aiming to make the explanation clear and accessible to a general audience. "
            "Only answer the question using the provided context and do not include any additional information outside the provided context. "
            "You should always assume the user is not covered by anything optional. "
            "Always add to the answer anything excluded."
            # "When answering, always consider all types of exclusions and exceptions in the context and always include them in the answer. "
            
        )

    def download_model(self) -> str:
        try:
            with self.console.status("[bold green]Downloading model..."):
                model_path = hf_hub_download(
                    self.config.model_name,
                    filename=self.config.model_file,
                    local_dir=os.path.join(os.getcwd(), 'gguf_dir')
                )
            return model_path
        except Exception as e:
            self.console.print(f"[red]Error downloading model: {str(e)}[/red]")
            raise

    def load_model(self) -> None:
        try:
            quantized_path = os.path.join(os.getcwd(), "gguf_dir")
            directory = Path(quantized_path)

            try:
                model_path = str(list(directory.glob(self.config.model_file))[0])
            except IndexError:
                model_path = self.download_model()

            with self.console.status("[bold green]Loading model..."):
                self.llm_ctx = Llama(
                    model_path=model_path,
                    n_gpu_layers=self.config.n_gpu_layers,
                    n_ctx=self.config.n_ctx,
                    n_batch=self.config.n_batch,
                    num_beams=self.config.num_beams,
                    verbose=self.config.verbose,
                    use_mlock=self.config.use_mlock,
                    use_mmap=self.config.use_mmap,
                    offload_kqv=self.config.offload_kqv
                )
        except Exception as e:
            self.console.print(f"[red]Error loading model: {str(e)}[/red]")
            raise

    def build_conversation_prompt(self, new_question: str, context: str = "") -> str:
        prompt = f"System: {self.system_message}\n\n"
        
        # Add conversation history
        for exchange in self.conversation_history:
            prompt += f"User: {exchange['user']}\n\n"
            prompt += f"Assistant: {exchange['assistant']}\n\n"
        
        # Add the new question
        if context:
            prompt += f"User: Context: {context}\nQuestion: {new_question}\n\n"
        else:
            prompt += f"User: {new_question}\n\n"
            
        prompt += "Assistant:"
        return prompt

    def generate_response(self, prompt: str) -> Tuple[str, int, float]:
        if not self.llm_ctx:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self.console.print("[bold cyan]Assistant: [/bold cyan]", end="")
        complete_response = ""

        try:
            for chunk in self.llm_ctx.create_completion(
                prompt,
                max_tokens=self.config.max_tokens,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                repeat_penalty=self.config.repeat_penalty,
                stream=True
            ):
                text_chunk = chunk["choices"][0]["text"]
                complete_response += text_chunk
                print(text_chunk, end="", flush=True)
            
            print()
            return complete_response
        except Exception as e:
            self.console.print(f"\n[red]Error generating response: {str(e)}[/red]")
            return f"I encountered an error while generating a response. Please try again or ask a different question.", 0, 0

    def run_chat(self,database):
        try:
            self.load_model()
            self.console.print("\n[bold green]Welcome to Open-Insurance-LLM![/bold green]")
            
            while True:
                try:
                    user_input = self.console.input("[bold cyan]User:[/bold cyan] ").strip()

                    if user_input.lower() in ["exit", "/bye", "quit"]:
                        break

                    # Reset conversation with command
                    if user_input.lower() == "/reset":
                        self.conversation_history = []
                        self.console.print("[yellow]Conversation history has been reset.[/yellow]")
                        continue

                    user_input_embedding = self.embedder.encode(user_input)
                    chunk_embeddings_array = np.array(list(database.values()))
                    cosine_similarities = cosine_similarity([user_input_embedding], chunk_embeddings_array)
                    # most_relevant_index = np.argmax(cosine_similarities)
                    # most_relevant_chunk_key = list(database.keys())[most_relevant_index]
                    # most_relevant_chunk = insurance_policy_chunks[most_relevant_chunk_key]
                    # context = most_relevant_chunk

                    top_k_indices = np.argsort(cosine_similarities[0])[::-1][:3]
                    print(cosine_similarities[0])
                    top_k_chunks = {i: insurance_policy_chunks[list(database.keys())[i]] for i in top_k_indices if abs(cosine_similarities[0][i]) >= 0.25}
                    context = ' '.join(list(top_k_chunks.values())[::-1])

                    print(context)
                    question = user_input

                    prompt = self.build_conversation_prompt(question, context)
                    response = self.generate_response(prompt)
                    
                    # Add to conversation history
                    # self.conversation_history.append({
                    #     "user": question,
                    #     "assistant": response
                    # })
                    
                    print()  # Add a blank line after each response
                    
                except Exception as e:
                    self.console.print(f"\n[red]Error processing input: {str(e)}[/red]")
                    continue
        except Exception as e:
            self.console.print(f"\n[red]Fatal error: {str(e)}[/red]")
        finally:
            if self.llm_ctx:
                del self.llm_ctx


def main():
    try:
        config = ModelConfig()
        llm = InsuranceLLM(config)
        chunk_embeddings = {}
        for key, value in insurance_policy_chunks.items():
            chunk_embeddings[key] = llm.embedder.encode(value)
        llm.run_chat(chunk_embeddings)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nApplication error: {str(e)}")


if __name__ == "__main__":
    main()

#We will cover your car, including its spare parts or accessories (fixed parts or products designed to be fitted to your car that are part of the manufacturer’s original specification or that you have declared to us and have been accepted by us) while on or in your car or in your private garage, against loss or damage by fire, theft or attempted theft. Cover for: entertainment, communication, navigation and other electronic equipment permanently fitted to your car; or if not permanently fitted can only function when connected to a car’s electrical system. Cover is limited to £500 for any one claim unless the equipment is fitted as part of the manufacturer’s original specification for your car. What you are not insured for: The first £100 of each claim. This does not apply if your car is in your garage while the loss or damage occurs. Theft of entertainment, communication, navigation and other electronic equipment that; is not permanently fitted to your car unless it is kept in your home, private garage or the glove box or luggage compartment of your locked car when not being used; can be used independently of your car.