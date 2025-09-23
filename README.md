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
cd src/llm
python3 llm_nodes.py
```

### run without docker
- On macOS, Docker cannot directly access hardware devices (microphone)
```
brew install portaudio
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd src/speech_to_text
python3 hello_tyson.py
```