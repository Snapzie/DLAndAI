import os
import time
from pathlib import Path
from llama_cpp import Llama
from rich.console import Console
from huggingface_hub import hf_hub_download
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class ModelConfig:
    '''
    Model configurations for the LLM. Most have been left as default settings
    '''
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
    '''
    InsuranceLLM is a Quantized Language Model building on the Llama architecture and has been finetuned
    for insurance-related queries and conversations.\n
    https://huggingface.co/Raj-Maharajwala/Open-Insurance-LLM-Llama3-8B-GGUF
    '''
    def __init__(self, config: ModelConfig):
        self.config = config
        self.llm_ctx = None
        self.console = Console()
        
        self.system_message = (
            "This is a chat between a user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. "
            # "The assistant should also indicate when the answer cannot be found in the context. "
            # "You are an expert from the Insurance domain with extensive insurance knowledge and "
            # "professional writer skills, especially about insurance policies. "
            # "Your name is OpenInsuranceLLM, and you were developed by Raj Maharajwala. "
            # "You are willing to help answer the user's query with a detailed explanation. "
            # "In your explanation, leverage your deep insurance expertise, such as relevant insurance policies, "
            # "complex coverage plans, or other pertinent insurance concepts. Use precise insurance terminology while "
            # "Use precise insurance terminology while "
            # "still aiming to make the explanation clear and accessible to a general audience. "
            # "Only answer the question using the provided context and do not include any additional information outside the provided context. "
            "You should always assume the user is not covered by anything optional. "
            # "Always add to the answer anything excluded. "
            # "If you are not provided a context, NEVER answer ANYTHING, politely tell the user you do not know the answer to ANY of the provided questions. "
            "If you are not provided a context always give the answer: 'I do not know the answer to this'. "
            "You can ONLY answer by stating facts from the provided context. You can NOT answer anything not mentioned in the provided context. "
            "You always answer the users questions politely if you can find information in the provided context. "
            "Never give advise or examples not provided in the context."
            
        )

    def download_model(self) -> str:
        '''
        Downloads the model in /gguf_dir directory. This only runs the first time the model is used.
        '''
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
        '''
        Loads and instantiates the model from /gguf_dir directory
        '''
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
    
    def generate_answer(self, new_question, context, metadata=None):
        '''
        Generates an answer to a prompt when provided a context. Optionally also outputs metadata
        surch as section and page numbers of provided context.
        \nInput:
        - new_question str: The embedded question the llm should answer
        - context str: The context the llm should base its answer in
        - metadata dict[str,str] optional: dict containing metadata about section and page of provided context
        \nreturns:
        - str, str: Answer to the provided question and a string representation of the provided metadata
        '''
        prompt = f"System: {self.system_message}\n\n"
        post_additions = 'Any exclusions or exceptions?' # Always added to all prompts
        prompt += f"User: Context: {context}\nQuestion: {new_question+post_additions}\n\n"
        prompt += "Assistant:"

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
            
            # Add information about metadata such as which section and which page context was found
            meta_string = ''
            if metadata:
                metadata_report = '\n\nThe information for this answer was found on the following sections and pages:\n'
                meta_string = '\n'.join([f'section: {meta_dict["section"]}, p. {meta_dict["page"]}' for meta_dict in metadata])
                print(metadata_report+meta_string, end="", flush=True)

            print()
            return complete_response,meta_string
        except Exception as e:
            self.console.print(f"\n[red]Error generating response: {str(e)}[/red]")
            return f"I encountered an error while generating a response. Please try again or ask a different question.", 0, 0