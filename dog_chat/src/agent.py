import os
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from middleware import NoInternalIDsMiddleware
from prompts import SYSTEM_PROMPT
from tools import search_dogs, get_dog_profile, get_litter_dogs, get_dog_titles, get_dogs_by_titles, search_show


def build_agent():
    # model = ChatOllama(
    #     model="qwen3:8b",
    #     temperature=0,
    #     num_ctx=16384
    # )

    # reviewer = ChatOllama(
    #     model="qwen3:8b",
    #     temperature=0,
    #     num_ctx=16384
    # )
    model = ChatOpenAI(model="gpt-6-luna",use_responses_api=True)
    reviewer = ChatOpenAI(model="gpt-6-luna",use_responses_api=True)


    return create_agent(
        model=model,
        tools=[
            search_dogs,
            get_dog_profile,
            get_litter_dogs,
            get_dog_titles,
            get_dogs_by_titles,
            search_show,
        ],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
        middleware=[
            NoInternalIDsMiddleware(reviewer_model=reviewer)
        ]
    )