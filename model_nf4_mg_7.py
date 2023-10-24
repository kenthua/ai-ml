import ray
import os
import transformers
import torch
from ray import serve
from fastapi import FastAPI
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, AutoConfig

# validator error - pip install pydantic==1.10.9

app = FastAPI()

model_id = "meta-llama/Llama-2-7b-chat-hf"

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 20, "num_gpus": 2})

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
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True
        )
           
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            token=os.environ["HUGGING_FACE_HUB_TOKEN"]
        )

        # prepare teh model wiht quantization and device map for multi-gpu
        model_nf4 = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=nf4_config,
            device_map="auto",
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
            max_length=200        
        )

        val = ""
        for seq in sequences:
             val = seq['generated_text']
             print("val: " + val)

        return val 

chat_app_nf4_mg = Chat.bind()
