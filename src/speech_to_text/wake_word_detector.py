import re
import time
import sys

import speech_recognition as sr


class WakeWordDetector:
    def __init__(self, pattern: str):
        self.wake_pattern = re.compile(pattern, re.IGNORECASE)
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    def normalize(self, text: str) -> str:
        # basic cleanup; keeps spaces/letters, strips most punctuation.
        return re.sub(r"[^\w\s]", "", text).strip()

    def heard_wake_pattern(self, text: str) -> bool:
        cleaned = self.normalize(text)
        return bool(self.wake_pattern.search(cleaned))
    
    def start_listening(self):
        print("Listening. Say 'Hey Tyson'…")
        while True:
            try:
                with self.microphone as source:
                    # Listen with short timeouts so the loop stays responsive
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)

                try:
                    # Google Web Speech API (free, requires internet)
                    text = self.recognizer.recognize_google(audio)
                    print(f"Heard: {text}", flush=True)
                except sr.UnknownValueError:
                    # Nothing intelligible
                    continue
                except sr.RequestError as e:
                    print(f"[STT request error: {e}]", file=sys.stderr, flush=True)
                    time.sleep(1)
                    continue

                if self.heard_wake_pattern(text):
                    # If you only want it to respond once and exit, uncomment:
                    return text
                # else: do nothing

            except sr.WaitTimeoutError:
                # No speech detected in the timeout window; just keep listening
                continue
            except KeyboardInterrupt:
                self.say("Exiting.")
                break

def main():
    from speech_synthesizer import SpeechSynthesizer

    detector = WakeWordDetector(r"\bhey\s+tyson\b")
    text = detector.start_listening()
    print(f"Detected wake word in: {text}")
    SpeechSynthesizer.say(text)

if __name__ == "__main__":
    main()
