import asyncio
import base64
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

SAMPLE_RATE = 24000  # match model output rate

# --- Speech capture ---
class SpeechDetector:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
    
    def listen_once(self):
        print("Listening for speech...")
        while True:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                try:
                    text = self.recognizer.recognize_google(audio)
                    print(f"Heard: {text}")
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    print(f"STT request error: {e}")
                    continue
                return text, audio
            except sr.WaitTimeoutError:
                continue
            except KeyboardInterrupt:
                print("Exiting.")
                return None, None

def audio_to_pcm16(audio_data, rate=SAMPLE_RATE):
    return audio_data.get_raw_data(convert_rate=rate, convert_width=2)

async def main():
    detector = SpeechDetector()
    client = AsyncOpenAI()

    async with client.realtime.connect(model="gpt-4o-realtime-preview-2024-10-01") as connection:
        # Configure session to use audio output
        await connection.session.update(
            session={
                "output_modalities": ["audio"],
                "model": "gpt-realtime",
                "type": "realtime",
            }
        )

        while True:
            # Capture speech
            text, audio_data = detector.listen_once()
            if audio_data is None:
                break

            # Convert to PCM16 bytes
            pcm16_bytes = audio_to_pcm16(audio_data)

            # Send audio to OpenAI
            await connection.input_audio_buffer.append(audio=base64.b64encode(pcm16_bytes).decode('utf-8'))
            await connection.input_audio_buffer.commit()

            # Ask assistant to respond with audio
            await connection.response.create()

            # Play assistant audio as it arrives
            # --- Smooth audio playback ---
            with sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32'
            ) as stream:
                async for event in connection:
                    if event.type == "response.output_audio.delta":
                        audio_bytes = base64.b64decode(event.delta)
                        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
                        audio_float = audio_np.astype(np.float32) / 32768.0
                        stream.write(audio_float)
                    elif event.type == "response.output_audio.done":
                        print("Audio response complete.")
                    elif event.type == "response.done":
                        print("--- Assistant response complete ---\n")
                        break

if __name__ == "__main__":
    asyncio.run(main())
