from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate




load_dotenv()


documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

vectorstore = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
)

retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
message = """
Answer the questions using the provided context. If the answer is not contained within the context, say "I don't know."
{question}
Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

chain = {"context":retriever, "question":RunnablePassthrough()} | prompt | llm

if __name__ == "__main__":
    response = chain.invoke( "What is the best pet for a beginner?")
    print(response)
    print(response.content)