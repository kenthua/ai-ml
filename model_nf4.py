import ray
from ray import serve
from fastapi import FastAPI
import os
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import transformers
import torch

# validator error - pip install pydantic==1.10.9

app = FastAPI()

model_id = "meta-llama/Llama-2-13b-chat-hf"

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 4, "num_gpus": 2})

@serve.ingress(app)
class Chat:
    tokenizer = ""
    pipeline = ""
    
    def __init__(self):
        # Load bnb4
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            token=os.environ["HUGGING_FACE_HUB_TOKEN"]
        )
        #self.tokenizer.pad_token = self.tokenizer.eos_token
        model_nf4 = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=nf4_config,
            token=os.environ["HUGGING_FACE_HUB_TOKEN"]
        )
        # is token needed for transfomers.pipeline
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_nf4,
            torch_dtype=torch.float16,
            device_map="auto",
            tokenizer=self.tokenizer,
            token=os.environ["HUGGING_FACE_HUB_TOKEN"]
        )

    @app.post("/")
    def chat(self, text: str) -> str:
        # Run inference
        print("text: " + text)
        sequences = self.pipeline(
            text,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=200,            
        )

        val = ""
        for seq in sequences:
             val = seq['generated_text']
             print("val: " + val)

        return val 

chat_app_nf4 = Chat.bind()