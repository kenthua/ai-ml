from ray import serve
from fastapi import FastAPI

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class MyFastAPIDeployment:
    @app.get("/hi")
    def say_hi(self) -> str:
        return "Hello world!"

app = MyFastAPIDeployment.bind()