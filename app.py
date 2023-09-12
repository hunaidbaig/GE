import os
import openai
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import ServiceContext
from llama_index.llms import OpenAI
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.node_parser import SentenceWindowNodeParser
import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
from langchain.vectorstores import FAISS

import openai

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]
DB_FAISS_PATH = "vectorstore/db_faiss"


st.header("Chat with the GE 💬 📚")

if "messages" not in st.session_state.keys():  # Initialize the chat message history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about GE Contract Documents",
        }
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(
        text="Loading and indexing the GE docs – hang tight! This should take 1-2 minutes."
    ):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()

        node_parser = SentenceWindowNodeParser.from_defaults(
            # how many sentences on either side to capture
            window_size=3,
            # the metadata key that holds the window of surrounding sentences
            window_metadata_key="window",
            # the metadata key that holds the original sentence
            original_text_metadata_key="original_sentence",
        )
        llm = OpenAI(model="gpt-4", temperature=0)
        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            ),
            node_parser=node_parser,
        )
        db = FAISS.from_documents(docs, service_context)
        db.save_local(DB_FAISS_PATH)

        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index


index = load_data()

# chat_engine = index.as_chat_engine(chat_mode="react", verbose=True)
query_engine = index.as_query_engine(
    similarity_top_k=4,
    # the target key defaults to `window` to match the node_parser's default
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)

if prompt := st.chat_input(
    "Your question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_engine.query(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history
