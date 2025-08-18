#Pytest-
import os

import pytest
import requests
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference
from langchain_openai import ChatOpenAI

#User Input -> Query
#Response -> Response
#Reference -> Ground Truth -> Expected Result
#Retrived context -> Top K retrieved docs

@pytest.mark.asyncio
async def test_context_precision():
    #Create object of a class for specific metric
    #Power of LLM + method metric -> score
    os.environ["OPENAI_API_KEY"] = "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"
    llm = ChatOpenAI(model ="gpt-4",temperature=0)
    langcain_llm = LangchainLLMWrapper(llm)
    context_precision = LLMContextPrecisionWithoutReference(llm=langcain_llm)
    question ="How many articles are there in the Selenium webdriver python course?"

    #Feed data

    responsedict = requests.post(
            url ="https://rahulshettyacademy.com/rag-llm/ask",
            json = {
                "question": "How many articles are there in the Selenium webdriver python course?",
                "chat_history": [
                    ]
            }
        ).json()

    print(responsedict)

    sample = SingleTurnSample(
        user_input= question,
        response=responsedict["answer"],
        retrieved_contexts=[responsedict["retrieved_docs"][0]["page_content"],
                            responsedict["retrieved_docs"][1]["page_content"],
                            responsedict["retrieved_docs"][2]["page_content"]]
        )

    score = await context_precision.single_turn_ascore(sample)
    print(score)
    assert score > 0.8