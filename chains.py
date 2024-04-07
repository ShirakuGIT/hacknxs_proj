
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import BedrockChat

from langchain_community.graphs import Neo4jGraph
import os
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)

from typing import List, Any
from utils import BaseLogger, extract_title_and_question
from langchain_google_genai import GoogleGenerativeAIEmbeddings

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
# Remapping for Langchain Neo4j integration
#os.environ["NEO4J_URL"] = url
neo4j_graph = Neo4jGraph(url=url, username=username, password=password)

from langchain_community.graphs import Neo4jGraph

def generate_embeddings_for_docs(neo4j_graph, embeddings):
    # Retrieve all document summaries from the Neo4j database
    documents = neo4j_graph.query("MATCH (doc:Document) RETURN doc.id AS id, doc.summary AS summary")

    # Iterate over the documents and generate embeddings
    for document in documents:
        doc_id = document['id']
        summary = document['summary']
        embedding = embeddings.embed_query(summary)

        # Ensure embedding is in list format if it's not already
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()

        # Directly incorporate the id and embedding into the query string
        update_query = f"""
            MATCH (doc:Document {{id: '{doc_id}'}})
            SET doc.embedding = {embedding}
        """
        neo4j_graph.query(update_query)



def load_embedding_model(embedding_model_name: str, logger=BaseLogger(), config={}):
    if embedding_model_name == "ollama":
        embeddings = OllamaEmbeddings(
            base_url=config["ollama_base_url"], model="llama2"
        )
        dimension = 4096
        logger.info("Embedding: Using Ollama")
    elif embedding_model_name == "openai":
        embeddings = OpenAIEmbeddings()
        dimension = 1536
        logger.info("Embedding: Using OpenAI")
    elif embedding_model_name == "aws":
        embeddings = BedrockEmbeddings()
        dimension = 1536
        logger.info("Embedding: Using AWS")
    elif embedding_model_name == "google-genai-embedding-001":        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        dimension = 384
        logger.info("Embedding: Using Google Generative AI Embeddings")
    else:
        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2", cache_folder="/embedding_model"
        )
        dimension = 384
        logger.info("Embedding: Using SentenceTransformer")
    return embeddings, dimension


def load_llm(llm_name: str, logger=BaseLogger(), config={}):
    if llm_name == "gpt-4":
        logger.info("LLM: Using GPT-4")
        return ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True)
    elif llm_name == "gpt-3.5":
        logger.info("LLM: Using GPT-3.5")
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)
    elif llm_name == "claudev2":
        logger.info("LLM: ClaudeV2")
        return BedrockChat(
            model_id="anthropic.claude-v2",
            model_kwargs={"temperature": 0.0, "max_tokens_to_sample": 1024},
            streaming=True,
        )
    elif len(llm_name):
        logger.info(f"LLM: Using Ollama: {llm_name}")
        return ChatOllama(
            temperature=0,
            base_url=config["ollama_base_url"],
            model=llm_name,
            streaming=True,
            # seed=2,
            top_k=10,  # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
            top_p=0.3,  # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
            num_ctx=3072,  # Sets the size of the context window used to generate the next token.
        )
    logger.info("LLM: Using GPT-3.5")
    return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)


def configure_llm_only_chain(llm):
    # LLM only response
    template = """
    You are a helpful assistant that helps a support agent with answering questions related to Indian Tax Laws.
    If you don't know the answer, just say that you don't know, you must not make up an answer.
    The context contains Indian Tax Laws and Documents and their relations and details.
    Prefer information that is directly related to the queried tax topics.
    Rely on the information from the document summaries and their corresponding entity relationships.
    If there is no information from your database regarding that particular tax law, DO NOT WRITE ABOUT IT.
    Instead, just say you don't know the answer, and tell I can help you in other tax laws that you have in your context.
    Standing instruction: If you don't have the exact information relating to the user's query, say you don't know.
    If you don't know the answer, just say so. Do not make up an answer.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    def generate_llm_output(
        user_input: str, callbacks: List[Any], prompt=chat_prompt
    ) -> str:
        chain = prompt | llm
        answer = chain.invoke(
            {"question": user_input}, config={"callbacks": callbacks}
        ).content
        return {"answer": answer}

    return generate_llm_output


def configure_qa_rag_chain(llm, embeddings, embeddings_store_url, username, password):
    # Define the system's context-aware prompt template
    general_system_template = """ 
    Use the following pieces of context to answer the question at the end.
    The context contains Indian Tax Laws and Documents and their relations and details.
    Prefer information that is directly related to the queried tax topics.
    Rely on the information from the document summaries and their corresponding entity relationships.
    If there is no information from your database regarding that particular tax law, DO NOT WRITE ABOUT IT.
    Instead, just say you don't know the answer, and tell I can help you in other tax laws that you have in your context.
    Standing instruction: If you don't have the exact information relating to the user's query, say you don't know.
    If you don't know the answer, just say so. Do not make up an answer.
    ----
    {summaries}
    ----
    """

    # Define the user's question template
    general_user_template = "Question:```{question}```"
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    # Load the QA with sources chain
    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=qa_prompt,
    )

    vector = generate_embeddings_for_docs(neo4j_graph, embeddings)

    # Vector + Knowledge Graph response
    kg = Neo4jVector.from_existing_index(
    embedding=embeddings,
    url=embeddings_store_url,
    username=username,
    password=password,
    database="neo4j",  # This should match your actual Neo4j database name
    index_name="documentSummaryEmbeddings",  # This should match your actual index name for tax-related documents
    text_node_property="summary",  # Assuming each document has a 'summary' field
    retrieval_query="""
MATCH (n)
OPTIONAL MATCH (n)-[r]->(m)
RETURN n, r, m

"""

    )


    kg_qa = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=kg.as_retriever(search_kwargs={"k": 10}),  # Adjust 'k' as needed to retrieve a reasonable number of documents
        reduce_k_below_max_tokens=False,
        max_tokens_limit=3375,
    )

    return kg_qa


def generate_ticket(neo4j_graph, llm_chain, input_question):
    # Get high ranked questions
    records = neo4j_graph.query(
        "MATCH (q:Question) RETURN q.title AS title, q.body AS body ORDER BY q.score DESC LIMIT 3"
    )
    questions = []
    for i, question in enumerate(records, start=1):
        questions.append((question["title"], question["body"]))
    # Ask LLM to generate new question in the same style
    questions_prompt = ""
    for i, question in enumerate(questions, start=1):
        questions_prompt += f"{i}. \n{question[0]}\n----\n\n"
        questions_prompt += f"{question[1][:150]}\n\n"
        questions_prompt += "----\n\n"

    gen_system_template = f"""
    You're an expert in formulating high quality questions. 
    Formulate a question in the same style and tone as the following example questions.
    {questions_prompt}
    ---

    Don't make anything up, only use information in the following question.
    Return a title for the question, and the question post itself.

    Return format template:
    ---
    Title: This is a new title
    Question: This is a new question
    ---
    """
    # we need jinja2 since the questions themselves contain curly braces
    system_prompt = SystemMessagePromptTemplate.from_template(
        gen_system_template, template_format="jinja2"
    )
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            SystemMessagePromptTemplate.from_template(
                """
                Respond in the following template format or you will be unplugged.
                ---
                Title: New title
                Question: New question
                ---
                """
            ),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )
    llm_response = llm_chain(
        f"Here's the question to rewrite in the expected format: ```{input_question}```",
        [],
        chat_prompt,
    )
    new_title, new_question = extract_title_and_question(llm_response["answer"])
    return (new_title, new_question)
