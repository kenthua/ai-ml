- if you get validator error
```
pip install pydantic==1.10.9
```

- on ray cluster
```
serve run model_nf4_mg:chat_app_nf4_mg
```

- on ray cluster
```
serve run config.yaml
```

- testing
```
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"text": "How do I bake a pie?"}'
```