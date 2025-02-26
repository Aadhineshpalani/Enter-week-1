# Enter-week-1
implrmentation of chatbot
import random
import nltk
from nltk.chat.util import Chat, reflections
from transformers import pipeline

# Download necessary NLTK data
nltk.download('punkt')

# Define chatbot responses using NLTK chat framework
pairs = [
    [
        r"hi|hello|hey",
        ["Hello! How can I assist you today?", "Hi there! How can I help?"]
    ],
    [
        r"(.*) your name?",
        ["I'm ImprovemBot, your AI assistant."]
    ],
    [
        r"how are you?",
        ["I'm just a bot, but I'm functioning as expected!"]
    ],
    [
        r"(.*) help (.*)",
        ["Sure! What do you need help with?"]
    ],
    [
        r"quit",
        ["Goodbye! Have a great day ahead!"]
    ]
]

# Initialize the chatbot with predefined responses
chatbot = Chat(pairs, reflections)

# Load a Transformer-based conversational model for more dynamic responses
nlp_model = pipeline("text-generation", model="facebook/blenderbot-400M-distill")

def get_response(user_input):
    user_input = user_input.lower()
    
    # Check if a predefined response exists
    response = chatbot.respond(user_input)
    if response:
        return response
    
    # If no predefined response, use NLP model
    generated_response = nlp_model(user_input, max_length=50, do_sample=True)
    return generated_response[0]['generated_text']

# Main loop for chatbot interaction
def chat():
    print("Hello! I'm ImprovemBot. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("ImprovemBot: Goodbye!")
            break
        response = get_response(user_input)
        print(f"ImprovemBot: {response}")

if __name__ == "__main__":
    chat()

