
try:
    import pyttsx3
    tts_engine = pyttsx3.init()
except Exception:
    tts_engine = None

class SpeechSynthesizer:
    
    @staticmethod
    def say(text: str):
        if tts_engine:
            tts_engine.say(text)
            tts_engine.runAndWait()
        print(text, flush=True)