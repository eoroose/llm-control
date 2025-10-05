## setup

.env
```
OPENAI_API_KEY=
```

### run without docker
- current Python version: 3.12.9
- On macOS, Docker cannot directly access hardware devices (microphone)
```
brew install portaudio
```

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

if you see the following warining:
```
UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
```
do:
```
pip install --upgrade webrtcvad-wheels
```

run main.py for real-time for speech-to-speech LLM interaction
```
python main.py
```