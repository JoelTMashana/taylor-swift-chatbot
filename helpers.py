from datetime import datetime
import sys
import time
from typing import Dict


def parse_retriever_input(params: Dict):
    return params["messages"][-1].content


def get_current_date():
    current_date = datetime.now()
    formatted_date = current_date.strftime("%B %d of the year %Y")
    return f"The current date is {formatted_date}"



def type_effect(message):
    for char in message:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.01)


def print_bot_message(message, is_instruction):
    print("\n---------------------------------------------------------------------")
    print("AI: ")
    if not is_instruction:
        type_effect(message)  
    else:
        sys.stdout.write(message)
    print()


def receive_user_input(bot_prompt, chat_history=False, is_instruction=False):
    
    print_bot_message(bot_prompt, is_instruction)
    
    if chat_history:
        chat_history.add_ai_message(bot_prompt)
    user_input = input("User: ")
    return user_input