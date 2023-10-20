import ray
from ray import serve
from fastapi import FastAPI
import os
from transformers import AutoTokenizer
import transformers
import torch

# validator error - pip install pydantic==1.10.9

app = FastAPI()

model = "meta-llama/Llama-2-7b-chat-hf"

@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0})

@serve.ingress(app)
class Chat:
    tokenizer = ""
    pipeline = ""
    
    def __init__(self):
        # Load model
        # self.model = pipeline("translation_en_to_fr", model="t5-small")
        # print("loaded model")
        self.tokenizer = AutoTokenizer.from_pretrained(model, token=os.environ["HF_API_TOKEN"])
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.float16,
            device_map="auto",
            token=os.environ["HF_API_TOKEN"]
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

        # for seq in sequences:
        #     print(f"Result: {seq['generated_text']}")

        return sequences

chat_app = Chat.bind()