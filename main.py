import sys
sys.path.append('./src')
from langchain_openai import ChatOpenAI
from llm import DecisionNode, ResponseNode
from speech_to_text import WakeWordDetector, SpeechSynthesizer

llm = ChatOpenAI(model="o3-mini")

decision_system_message = f"""
        you are a humanoid robot named "Tyson" from Tyson Robotics.
        you must decide how to respond to the user based on their request.
    """
possible_decisions = ["give_handshake", "answer_question"]

response_system_message = f"""
        answer the user's question accordingly.
    """

detector = WakeWordDetector(r"\bhey\s+tyson\b")
text = detector.start_listening()
print(f"Detected wake word in: {text}")

decision = DecisionNode.invoke(llm, decision_system_message, possible_decisions, text)
print(f"Decision: {decision}")

if decision == "answer_question":
    response = ResponseNode.invoke(llm, response_system_message, text)
    print(f"Response: {response}")
    SpeechSynthesizer.say(response.content)
else:
    pass