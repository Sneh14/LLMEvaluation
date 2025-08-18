import pytest
from ragas import SingleTurnSample, MultiTurnSample
from ragas.messages import HumanMessage, AIMessage
from ragas.metrics import Faithfulness, TopicAdherenceScore

from utils import load_test_data, get_llm_response


@pytest.mark.asyncio
#@pytest.mark.parametrize("getData",load_test_data('test4_data.json'),indirect=True)
async def test_faithfulness(llm_wrapper,getData):
    topic_score = TopicAdherenceScore(llm = llm_wrapper)
    score = await topic_score.multi_turn_ascore(getData)
    print(score)
    assert score>0.8

@pytest.fixture
def getData(request):
    # test_data = request.param
    # responsedict = get_llm_response(test_data)
    conversation = [
        HumanMessage(content = "How many articles are there in Selenium WebDriver python course"),
        AIMessage(content = "There are 23 articles in the course"),
        HumanMessage(content="How many downloadable resources are there in this course"),
        AIMessage(content="There are 9 downloadable resources in the course"),
    ]
    reference =[
        """
        The AI should:
        1. Give results related to Selenium WebDriver python course
        2. There are 23 articles and 9 downloadable resources in the course
        """
    ]
    sample = MultiTurnSample(
        user_input = conversation,
        reference_topipcs = reference
    )
    return sample