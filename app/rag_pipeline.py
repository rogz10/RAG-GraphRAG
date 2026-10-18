from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.config import settings
from app.vector_store import get_vector_store

SYSTEM_PROMPT = """
Tu es un assistant RAG précis et fiable
Réponds uniquement avec les informations présentes dans le contexte
Si la réponse n'est pas dans le contexte indique que tu ne sais pas
Cite les passages utilisés avec leur identifiant comme [source 1]

Contexte
{context}
"""


def build_rag_chain():
    """construit la chaîne rag complète"""
    llm = ChatOllama(model=settings.llm_model)
    retriever = get_vector_store().as_retriever(
        search_kwargs={"k": settings.retrieval_k},
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ],
    )

    answer_chain = prompt | llm | StrOutputParser()
    return retriever, answer_chain


def ask_question(question: str):
    """pose une question au pipeline rag"""
    retriever, answer_chain = build_rag_chain()
    documents = retriever.invoke(question)
    context = "\n\n".join(
        f"[source {index}]\n{document.page_content}"
        for index, document in enumerate(documents, start=1)
    )
    answer = answer_chain.invoke(
        {
            "context": context,
            "input": question,
        },
    )

    return {
        "answer": answer,
        "sources": format_sources(documents),
    }


def format_sources(documents):
    """prépare les sources pour la réponse api"""
    return [
        {
            "id": index,
            "source": document.metadata.get("source"),
            "page": document.metadata.get("page"),
            "start_index": document.metadata.get("start_index"),
            "passage": document.page_content,
        }
        for index, document in enumerate(documents, start=1)
    ]
