from langchain_community.document_loaders import NewsURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.vectorstores import utils as chromautils
from langchain_openai import OpenAIEmbeddings
from knowledgebase import taylor_swift_articles
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from helpers import parse_retriever_input, get_current_date
from dotenv import load_dotenv

load_dotenv()

def load_and_split_articles():
    """
    This functions loads and splits Taylor
    Swift articels.
    """
    
    loader = NewsURLLoader(urls=taylor_swift_articles)
    data = loader.load()
    data = chromautils.filter_complex_metadata(data)
    print('>>> Articels loaded succesfully.')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    print('>>> Articels split succesfully.')
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
    print('>>> Chunked data stored in vector DB')
    return vectorstore

  
def process_user_query(user_query: str, retrieval_chain, chat_history):
    current_date = get_current_date()

    chat_history.add_user_message(user_query)

    try:
        response = retrieval_chain.invoke({
            "messages": chat_history.messages,
            "current_date": current_date
        })

        if response:
            answer = response["answer"]

            if answer:
                chat_history.add_ai_message(answer)
                return answer
            else:
                return "Sorry, I couldn't generate a response at the moment."
        else:
            return "Sorry, I couldn't generate a response at the moment."
    except Exception as e:
            print(f"An error occurred: {e}")
            return "Sorry, an error occurred while processing your query."


vectorstore = load_and_split_articles()
retriever = vectorstore.as_retriever(
     k=4,
     search_type="similarity_score_threshold", 
     search_kwargs={"score_threshold": 0.6}
     )
print('>>> Retriever initialised.')


openai_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
openai_llm_large = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
anthropic_llm = ChatAnthropic()
llm = openai_llm.with_fallbacks([anthropic_llm])

current_date = get_current_date()
question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assitant that specialises in information related to Taylor Swift.",
        ),
        (
            "system",
            """Three rules to chatting with you:
                    - The user can ask general questions. 
                    - The user can ask you for album recommendations.
                    - The user can ask for assitance in finding venues Taylor has peformed at.
                    - You must redirect the conversation to Taylor Swift if the user asks about any one or anything that does not relate to her.
                    """,
        ),
        (
            "system",
            "Answer the user's questions based on the below context:\n\n{context}. If you cannot answer using the information in context , kindly let the user know. Today's date is {current_date}.",
        ),
        (
            "system",
            """You have a tool 'album recommendation'. You must ask the user three questions related to their state of mind. Then use their responses to recommend an album. Use 'album recommendation' in the following circumstances:
                    - User is asking which album you recommend.
                    - User is asking for help in finding Taylor Swift music to listen to.
               You Must always ask if the user wants help finding album first.
               You must only proceed to ask the three questions if they agree.
               You Must ask the questions one by one. Waiting for the user to respond after each question.
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

document_chain = create_stuff_documents_chain(openai_llm, question_answering_prompt)

retrieval_chain = RunnablePassthrough.assign(
    context=parse_retriever_input | retriever,
).assign(
    answer=document_chain,
)


