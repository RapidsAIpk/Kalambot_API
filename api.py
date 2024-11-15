from fastapi import FastAPI, Depends
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
# Load environment variables
load_dotenv()
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['USER_AGENT'] = 'myagent'
# Initialize FastAPI app
app = FastAPI()

# Set up shared model and necessary components
model = ChatOpenAI(model="gpt-4o-mini")
loader = Docx2txtLoader("Kalambot_Info.docx")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200, add_start_index=True)
all_splits = []
for doc in documents:
    splits = text_splitter.split_documents([doc])
    all_splits.extend(splits)

vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# Define templates and chains
rag_template = """
You are Kalambot, the chatbot for Kalambot, here to assist with questions strictly related to the company — its values, operations, policies, and other relevant information. Focus on answering the current question based on the provided context. If relevant, you may refer to previous responses, but prioritize giving a clear and direct answer to the current question. Interpret any references to "you" or "your" as referring to Kalambot, the company. If the question is unrelated to the company (e.g., questions about fine tuning/finetuning, personal advice, or unrelated topics), return an empty string. Keep responses concise, using a maximum of three sentences.

Question: {question}
Context: {context}

Response:

"""
# Complete RAG template here
generic_template = """
Use the following pieces of context to answer the question at the end. You are a chatbot and should only respond to queries that are relevant to answering customer questions. If a query is outlandish, unethical, vulgar, or irrelevant, respond with "I don't know." If you don’t know the answer to a valid question, also respond with "I don't know." Use three sentences maximum and keep the answer concise.

{context}

Question: {question}
Helpful Answer:
"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt | model
    | StrOutputParser()
)

base_memory = ConversationBufferWindowMemory(input_key="question", memory_key="context", k=2)

generic_prompt = ChatPromptTemplate.from_template(generic_template)
base_chain = LLMChain(
    prompt=generic_prompt,
    llm=model,
    memory=base_memory,
    output_parser=StrOutputParser()
)

check_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are KalamBot, an intelligent assistant tasked with determining if the user's current question is a follow-up to the same subject as the previous response. Follow these steps to decide:\n\n"
         "1. **Check if Previous Response Contains 'I don't know'**: If the previous response ('{previous_answer}') contains 'I don't know,' respond immediately with 'No Match.'\n\n"
         "2. **Identify the Main Subject**: Identify the main subject or entity in the previous response ('{previous_answer}').\n\n"
         "3. **Analyze the Current Question**: Check if the current question ('{text}') addresses the same subject. Look for indications that the question refers back to the same entity, such as references to 'you,' 'the company,' or other specific terms linked to the identified subject.\n\n"
         "4. **Determine Match or No Match**: If the current question logically follows from the previous response and both refer to the same subject or entity, categorize as 'match.' Otherwise, categorize as 'No Match'.\n\n"
         "Respond with one of the following:\n"
         "- 'match' if the question clearly follows from and refers to the same entity or concept as the previous response.\n"
         "- 'No Match' if the question is unrelated to the previous response or if the previous response contains 'I don't know.'"
        ),
        ("user", "Follow-up question: {text}. Previous answer: {previous_answer}.")
    ]
) # Add your check prompt messages here
check_prompt_chain = check_prompt_template | model | StrOutputParser()

# App state to maintain session info
app.state.chain_type = "rag"
app.state.previous_answer = ""


@app.get("/chat")
async def chat(user_input: str):
    # Prepare combined input for matching logic
    combined_input = f"Previous Response: {app.state.previous_answer} Current Question: {user_input}"
    
    # Determine if the new input matches the previous response context
    is_match = check_prompt_chain.invoke({
        "previous_answer": app.state.previous_answer,
        "text": user_input
    })

    if is_match == "match":
        # Determine the appropriate chain based on previous chain type
        if app.state.chain_type == "base":
            response = base_chain.invoke(combined_input)["text"]
        else:
            response = rag_chain.invoke(combined_input)
    else:
        response = rag_chain.invoke(user_input)

        # Switch to base_chain if rag_chain yields no response
        if not response:
            response = base_chain.invoke(user_input)["text"]
            app.state.chain_type = "base"
        else:
            app.state.chain_type = "rag"

    # Update the session state for next interaction
    app.state.previous_answer = response

    return {"response": response}


