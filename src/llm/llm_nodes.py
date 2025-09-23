from typing import List, Dict, Annotated, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()


class DecisionNode():

    @staticmethod
    def invoke(llm: ChatOpenAI, system_message: str, decisions: List[str], question: str) -> str:
        # Define the schema for the decision output -> this will ensure the LLM responds in a structured way
        # the response must be one of the provided decisions
        decision_output_schema = {
            "title": "DecisionOutput",
            "description": "A decision output containing the chosen action",
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": decisions,
                    "description": "the decision/action that must be taken based on the request"
                }
            },
            "required": ["decision"]
        }
        
        query_prompt_template = ChatPromptTemplate([
            ("system", system_message),
            ("user", question),
        ])

        structured_llm = llm.with_structured_output(decision_output_schema)
        return structured_llm.invoke(query_prompt_template.format_messages())['decision']
    
class ResponseNode():
    
    @staticmethod
    def invoke(llm: ChatOpenAI, system_message: str, question: str) -> str:
        prompt = ChatPromptTemplate([
            ("system", system_message),
            ("user", question),
        ])
        return llm.invoke(prompt.format_messages())


if __name__ == "__main__":
    llm = ChatOpenAI(model="o3-mini")
    system_message = f"""
        you are a humanoid robot named "Tyson" from Tyson Robotics.
        you must decide how to respond to the user based on their request.
    """
    possible_decisions = ["give_handshake", "answer_question"]
    prompt = "can you give me a handshake?"
    decision = DecisionNode.invoke(llm, system_message, possible_decisions, prompt)
    print(decision)

