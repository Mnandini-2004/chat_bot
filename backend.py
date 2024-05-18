import os
import bs4
import getpass
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

def response(user_query):
   
    loader = WebBaseLoader(
        web_paths=('https://en.wikipedia.org/wiki/Natural_language_processing',),
    )

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    cohere_api_key = "5cCrtFbe3SSTS0sOXx7bxsqRpS0Csyp1YzU5BvHN"
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0",cohere_api_key=cohere_api_key)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    groq_api_key="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"
    chat = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-70b-8192")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    template = """Use the following pieces of context to answer the question at the end.
              Say that you don't know when asked a question you don't know, donot make up an answer. Be precise and concise in your answer.

             {context}

             Question: {question}

             Helpful Answer:"""

    # Add the context to your user query
    custom_rag_prompt = PromptTemplate.from_template(template)   
    retriever = vectorstore.as_retriever(search_type="similarity")
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
        {"context": retriever , "question": RunnablePassthrough()}
        | custom_rag_prompt
        | chat
        | StrOutputParser()
    )

    return rag_chain.invoke(user_query) 

'''import os
import bs4
import getpass
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
def response(user_query):
    loader=PyPDFLoader("/content/dataset_new.pdf")
    docs1=loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs1)
    cohere_api_key = "leTByPB6J9FNbFIup99z08dhPaFwiquAlRqScvJv"
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0",cohere_api_key=cohere_api_key)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    groq_api_key="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"
    chat = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    template = """Use the following pieces of context to answer the question at the end.
            Say that you don't know when asked a question you don't know, donot make up an answer. Be precise and concise in your answer.

            {context}

            Question: {input}

            Helpful Answer:"""

    # Add the context to your user query
    prompt = PromptTemplate.from_template(template)
    from langchain.chains.combine_documents import create_stuff_documents_chain
    document_chain=create_stuff_documents_chain(chat,prompt)
    retriever=vectorstore.as_retriever()
    from langchain.chains import create_retrieval_chain
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    user_query="Who is the HOD of CS department"

    res=retrieval_chain.invoke({"input":user_query})

    return res['answer']'''
