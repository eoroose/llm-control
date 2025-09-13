from typing import List, Annotated, Literal
from typing_extensions import TypedDict
from IPython.display import Image, display
import requests
import json
import pandas as pd

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from langgraph.graph import START, StateGraph

import time
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, TypeVar

# For logging and visualization
import logging
from IPython.display import display, Markdown, HTML
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage

import asyncio


class State(TypedDict):
    question: str
    decision: str
    response: str

class DecisionOutput(TypedDict):
    """Generate a decision based on the request"""
    decision: Annotated[
        Literal["give_handshake", "answer_question"],
        "the decision/action that must be taken based on the request"
    ]

system_message = f"""
        you are a humanoid robot named "Tyson" from Tyson Robotics.
        you must decide how to respond to the user based on their request.
    """

user_prompt = "command: {input}"

query_prompt_template = ChatPromptTemplate([
    ("system", system_message), 
    ("user", user_prompt)
])

llm = ChatOpenAI(model="o3-mini")

def decide(state: State) -> DecisionOutput:
    """make a decision based on the request"""
    prompt = query_prompt_template.invoke(
        {
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(DecisionOutput)
    result = structured_llm.invoke(prompt)
    return {"decision": result["decision"]}


def decision_tree(state: State):
    if state["decision"] == "give_handshake":
        response = "handshake given"
    elif state["decision"] == "answer_question":
        prompt = query_prompt_template.invoke(
            {
                "input": state["question"],
            }
        )
        response = llm.invoke(prompt)
    return {"response": response}

graph_builder = StateGraph(State).add_sequence(
        [decide, decision_tree]
    )
graph_builder.add_edge(START, "decide")
graph = graph_builder.compile()

if __name__ == "__main__":

    async def main():
        async for chunk in graph.astream({"question": "what is your name and who do you work for?"}):
            if "decision_tree" in chunk:
                print(chunk['decision_tree']['response'].content)
            else:
                print(chunk)
                

    asyncio.run(main())
