from helpers import receive_user_input
from langchain_community.tools import GooglePlacesTool


def call_google_places_api_with_venue(chat_history):
    places = GooglePlacesTool()   
    venue = receive_user_input("Which venue are you interested in finding?", chat_history=chat_history)
    chat_history.add_user_message(venue)
    
    try:
        venue_details = places.run(venue)
        chat_history.add_ai_message(venue_details)
        return venue_details
    except Exception as e:
            print(f"An error occurred: {e}")
            return "Sorry, an error occurred while processing your query." 