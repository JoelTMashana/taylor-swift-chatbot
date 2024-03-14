from langchain.memory import ChatMessageHistory
from helpers import print_bot_message, receive_user_input
from RAG import retrieval_chain, process_user_query
from task_oriented_dialogue import  call_google_places_api_with_venue

def begin_chat():
    chat_history = ChatMessageHistory()
    
    user_input = 'Tell me what you do? What types of questions can I ask you?'
    response = process_user_query(user_input, retrieval_chain, chat_history)
    print_bot_message(response, is_instruction=False)
    
    while True:
        user_input = receive_user_input("Enter your query (type 'exit' to end):", chat_history=False, is_instruction=True)
        if user_input.strip().lower() == "exit":
            break

        elif user_input.strip().lower() == "find venue":
            user_input = "List some stadiums or venues Taylor Swift has taken her most recent tour so far."
            response = process_user_query(user_input, retrieval_chain, chat_history)
            print_bot_message(response, is_instruction=False)
            venue_details = call_google_places_api_with_venue(chat_history) 
            print_bot_message(venue_details, is_instruction=False)
        else:
            response = process_user_query(user_input, retrieval_chain, chat_history)
            print_bot_message(response, is_instruction=False)

