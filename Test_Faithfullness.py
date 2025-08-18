import pytest
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness

from utils import load_test_data, get_llm_response


@pytest.mark.asyncio
@pytest.mark.parametrize("getData",load_test_data('test4_data.json'),indirect=True)
async def test_faithfulness(llm_wrapper,getData):
    faithful = Faithfulness(llm = llm_wrapper)
    score = faithful.single_turn_ascore(getData)
    print(score)
    assert score>0.8

@pytest.fixture
def getData(request):
    test_data = request.param
    responsedict = get_llm_response(test_data)
    sample = SingleTurnSample(
        user_input=responsedict["question"],
        response = responsedict["answer"],
        retrieved_contexts=[doc["page_content"] for doc in responsedict["retrieved_docs"]]
    )