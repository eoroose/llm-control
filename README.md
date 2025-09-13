## setup

.env
```
OPENAI_API_KEY=
```

```
docker compose up -d --build
```

run test
```
docker compose exec -it llm-control /bin/bash
cd src
python3 llm_control.py
```