import pytest
from langsmith import evaluate
from ragas import SingleTurnSample, EvaluationDataset
from ragas.metrics import ResponseRelevancy, FactualCorrectness

from utils import load_test_data, get_llm_response

@pytest.mark.parametrize("getData",load_test_data('test5_data.json'),indirect=True)
@pytest.mark.asyncio
async def test_relevancy_factual(llm_wrapper,getData):
    metrics = [ResponseRelevancy(llm=llm_wrapper),
               FactualCorrectness(llm=llm_wrapper)]

    eval_dataset = EvaluationDataset([getData])
    results = evaluate(dataset = eval_dataset, metrics = metrics)
    print(results)



@pytest.fixture
def getData(request):
    test_data = request.param
    responsedict = get_llm_response(test_data)
    sample = SingleTurnSample(
        user_input=test_data["question"],
        response=responsedict["answer"],
        retrieved_contexts=[doc["page_content"] for doc in responsedict["retrieved_docs"]],
        reference = test_data["reference"]
    )
    return sample