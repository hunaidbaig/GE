import os
import openai
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import ServiceContext
from llama_index.llms import OpenAI
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.node_parser import SentenceWindowNodeParser
import pandas as pd
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall,
)
from ragas.metrics.critique import harmfulness
from ragas.llama_index import evaluate

# attach to the same event-loop
import nest_asyncio

nest_asyncio.apply()


os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]
# create the sentence window node parser w/ default settings
# creates nodes with spilitted document info and metadata info stored in each node.
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
# Initializing OpenAI gpt-4 model with 0 temperature
llm = OpenAI(model="gpt-4", temperature=0)
# setting up service context with nodes and llm
# embeddings are done with huggingface embeddings model
ctx = ServiceContext.from_defaults(
    llm=llm,
    embed_model=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ),
    node_parser=node_parser,
)
# loading the directory
documents = SimpleDirectoryReader("data").load_data()
sentence_index = VectorStoreIndex.from_documents(documents, service_context=ctx)
# Vector Store is a dedicated storage system that houses embeddings and their corresponding textual data, ensuring quick and efficient retrieval.
query_engine = sentence_index.as_query_engine(
    similarity_top_k=2,
    # the target key defaults to `window` to match the node_parser's default
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)
print(query_engine.query('What is the purpose of this procedural document?'))
eval_questions = [
    "What is the purpose of this procedural document?",
    "What entities does this procedural document apply to?",
    "What is the purpose of due diligence according to this document?",
    "What is the scope of the due diligence requirements outlined in this policy?"
    "Who is the target audience for this procedural document?",
    "What are the responsibilities of the Business \/ Commercial Team\/Channel in relation to ITPs?",
    "What role does the Commercial Approver play in the process?",
]

eval_answers = [
    "The purpose of this procedural document is to establish the guidelines for engaging with ITPs and to ensure alignment with ALC and GE standards. It outlines the criteria to form contractual relationships and emphasizes the importance of avoiding relationships that could damage GE\u2019s reputation or violate laws.",
    "This document applies to all ITPs, including their sub dealers\/contractors, and shorter term indirect relationships. However, there is an exception for cases where country\/region laws prevent ALC from performing due diligence on sub dealers\/contracts.",
    "The purpose of due diligence is to assess the risk associated with engaging with a commercial party before entering into a binding relationship. Non-binding quotes do not require due diligence, but they still go through a quoting process.",
    "The scope of the due diligence requirements includes ITPs with whom GE plans to establish a binding relationship. This document does not cover Direct Party relationships. For more information, refer to Appendix A.",
    "The intended audience for this procedural document is the Commercial Teams and Channel Excellence Managers. Compliance-related procedures, such as due diligence, will be detailed in a separate Compliance Team document.",
    "The Business \/ Commercial Team\/Channel is responsible for various tasks related to ITPs, including being the primary contact for ITP Excellence\/ITP Manager, gathering documentation for onboarding, conducting Commercial Needs Assessments, entering information in SFDC, performing needs assessment strategies, notifying Regional Compliance Team of changes, handling red flags or triggering events, confirming sub dealers' due diligence, providing due diligence information, setting up vendors as ITPs, managing payments, working on renewals and amendments, and offboarding ITPs.",
    "The Commercial Approver reviews requests in SFDC, communicates with the Commercial Team\/Channel Excellence and Compliance regarding onboarding issues, and plays a role in the approval process.",
]
eval_answers = [[a] for a in eval_answers]
metrics = [
    faithfulness,
    answer_relevancy,
    context_relevancy,
    harmfulness,
    context_recall,
]
result = evaluate(query_engine, metrics, eval_questions, eval_answers)
print(result)
# # Question will be given in query
# window_response = query_engine.query("Summarize Appendix D for me")
# print(window_response)