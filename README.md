
# LLM Evaluation Framework (RAG-based)

This repository provides a testing and evaluation framework for Custom Large Language Models (LLMs) built on Retrieval-Augmented Generation (RAG) architecture.

It enables structured, metric-driven evaluation of LLM performance across retrieval, augmentation, and generation modules, ensuring reliability, factual accuracy, and conversational consistency.




## Author

- [@Snehal](https://github.com/Sneh14)
- emailAddress : snehalp59@gmail.com


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/Sneh14)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/snehal-patil-69654953/)


## Features

- Custom LLM Architecture Evaluation
    - Metrics: context precision, recall, faithfulness, factual correctness, response   relevancy, topic adherence, rubric scores.
- End-to-End Testing Scope
    - Covers retrieval, augmentation, and generation modules.
- Multi-Conversation Scenarios
    - Benchmarks LLM performance in multi-turn dialogues.
- Synthetic Test Data
    - Generate and leverage synthetic datasets for evaluation.
- Pytest Integration
    - Optimized for pytest framework to ensure efficient and standardized test runs.

## Evaluation Metrics

The framework evaluates models by:
- Retrieving Top-K relevant documents from a vector database.
- Generating responses using original queries and retrieved context.
- Scoring responses across multiple dimensions:
- Precision & Recall
- Faithfulness & Factual Correctness
- Relevance & Topic Adherence
- Rubric-based Scoring


## Tech Stack

- Programming Language: Python 3.9+
- Frameworks & Tools:
    - Pytest – structured and efficient test execution
    - LangChain – orchestration of RAG components

- Evaluation & Metrics:
    - Custom metric scripts for precision, recall, faithfulness, relevance, rubric scores

- Synthetic Data Generation:
    - LLM-based data synthesis for evaluation datasets

## Installation

**Clone the Repository:**

git clone https://github.com/Sneh14/LLMEvaluation.git

cd LLMEvaluation



