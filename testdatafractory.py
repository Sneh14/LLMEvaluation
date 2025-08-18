from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
import nltk
from conftest import llm_wrapper

nltk.data.path.append("C:/Users/Snehal/OneDrive/Documents/LLM_Eval/nltk")
llm=ChatOpenAI(model="gpt-4", temperature=0)
llm_langchain=LangchainLLMWrapper(llm)
# to convert data in to vector format we will create object of OPenAIEmbeddings class
embed=OpenAIEmbeddings()

# Class to load files from local directory so that Ragas can fetch document from this directory and scan to generate test cases
loader=DirectoryLoader(
    path="C:/Users/Snehal/OneDrive/Documents/LLM_Eval/fs11",  # Path of the directory where files are stored
    glob='**/*.docx',  # what types of files needs to be scanned
    loader_cls=UnstructuredWordDocumentLoader
    # As we dont whats the format of word doc so we will use UnstructuredWordDocumentLoader class
)
docs=loader.load()  # to laod all document objects
# this object belongs to Langchain family and Ragas wont understand it
# So to make it understandable for Ragas we will create object of embeddings wrapper
# we created this object so that ragas can scan all embedded documents
generate_embeddings=LangchainEmbeddingsWrapper(embed)
generator=TestsetGenerator(llm=llm_langchain, embedding_model=generate_embeddings)
dataset=generator.generate_with_langchain_docs(docs, testset_size=20)
print(dataset.to_list())
