import os
import ray
import transformers
import torch
from ray import serve
from fastapi import FastAPI
from transformers import AutoTokenizer

# validator error - pip install pydantic==1.10.9

## multi-gpu, single-host

app = FastAPI()

model = "meta-llama/Llama-2-13b-chat-hf"

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 4, "num_gpus": 2})

@serve.ingress(app)
class Chat:
    tokenizer = ""
    pipeline = ""
    
    def __init__(self):
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            token=os.environ["HUGGING_FACE_HUB_TOKEN"]
        )
        # is token needed for transfomers.pipeline
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.float16,
            device_map="auto",
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

chat_app = Chat.bind()
