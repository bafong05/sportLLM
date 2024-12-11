"""
A terminal-based AI chatbot implementation using LangChain.

Used kaggle lego database to make a more specific chatbot
"""

import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Set up your OpenAI API key
os.environ["OPENAI_API_KEY"] = "add key"

# Initialize the OpenAI model
model = ChatOpenAI(model="gpt-4o-mini")


# Function to load and preprocess LEGO dataset
DATASET_FOLDER = 'lego-database'  # Update this to your actual folder path


# Function to load and preprocess LEGO dataset
def load_data():
    print("Loading LEGO dataset...")
    parts = pd.read_csv(os.path.join(DATASET_FOLDER, "parts.csv"))
    inventory_parts = pd.read_csv(os.path.join(DATASET_FOLDER, "inventory_parts.csv"))

    # Merge datasets
    data = inventory_parts.merge(parts, on="part_num")
    return data


# Chatbot logic
def chatbot():
    print(
        "Welcome to the LEGO Set Recommender! Describe your pieces, and I'll suggest sets. Type 'exit' to end the conversation.\n")
    data = load_data()

    messages = [
        SystemMessage(
            content="You are a LEGO assistant. Interpret user input about their LEGO pieces and suggest sets from the dataset.")
    ]

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            print("Goodbye!")
            break

        # Add user input to the conversation
        messages.append(HumanMessage(content=user_input))

        # Use OpenAI to process user input and extract useful details
        extraction_prompt = f"""
        The user has described some LEGO pieces or their characteristics: "{user_input}". 
        Extract relevant part IDs or descriptions in a structured way.
        """
        extraction_response = model.invoke([HumanMessage(content=extraction_prompt)]).content

        # Use OpenAI to decide on recommendations
        recommendation_prompt = f"""
        Based on the following extracted user input: "{extraction_response}" and this dataset structure:
        {data.head(5).to_string(index=False)}
        Recommend the top LEGO sets and explain why they match.
        """
        ai_response = model.invoke([HumanMessage(content=recommendation_prompt)]).content

        # Print the AI's response
        print("AI:", ai_response)

        # Add AI response to conversation history
        messages.append(AIMessage(content=ai_response))


# Run the chatbot
if __name__ == "__main__":
    chatbot()
