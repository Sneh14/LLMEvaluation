import os

import pytest
import requests
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall

@pytest.mark.asyncio
async def test_context_recall():
    llm = ChatOpenAI(model="gpt-4",temperature=0)
    langcgain_llm = LangchainLLMWrapper(llm)
    context_recall = LLMContextRecall(llm=langcgain_llm)
    os.environ[
        "OPENAI_API_KEY"] = "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"

    responsedict = requests.post(
        url="https://rahulshettyacademy.com/rag-llm/ask",
        json={
            "question": "How many articles are there in the Selenium webdriver python course?",
            "chat_history": [
            ]
        }
    ).json()

    print(responsedict)

    sample = SingleTurnSample(
        user_input = "How many articles are there in the Selenium webdriver python course?",
        retrieved_contexts = [responsedict["retrieved_docs"][0]["page_content"],
                            responsedict["retrieved_docs"][1]["page_content"],
                            responsedict["retrieved_docs"][2]["page_content"]],
        reference = "23"
    )

    score = await context_recall._single_turn_ascore(sample)
    print(score)