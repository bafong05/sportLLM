"""
A terminal-based AI chatbot implementation using LangChain.

debigging check
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Set up your OpenAI API key to be able to use openAI's API
os.environ["OPENAI_API_KEY"] = "replace with the API key!"

# Initialize the model. gpt 4o is the most recent, but it changes (sometimes they get out of date)
model = ChatOpenAI(model="gpt-4o-mini")

# Define the workflow-- basically define a new graph that keeps messages in memory
workflow = StateGraph(state_schema=MessagesState)

# Function that calls the model
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

# Add the node to the workflow
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory to persist conversation history
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Terminal Chatbot Logic
def chatbot():
    print("Welcome to this little AI Chatbot! Type 'exit' to end the conversation.\n")
    language = "English"
    config = {"configurable": {"thread_id": "terminal-chat"}}
    messages = [
        SystemMessage(content="You are a helpful assistant. Answer concisely and accurately.")
    ]

    while True:
        user_input = input("You: ")
        # For user exiting
        if user_input.strip().lower() == "exit":
            print("Goodbye!")
            break

        # Add user input to the conversation
        messages.append(HumanMessage(content=user_input))

        # Stream the AI's response
        print("AI: ", end="", flush=True)
        ai_response = ""
        for chunk, metadata in app.stream(
            {"messages": messages, "language": language},
            config,
            stream_mode="messages",  # Stream AI responses token-by-token
        ):
            if isinstance(chunk, AIMessage):
                print(chunk.content, end="", flush=True)
                ai_response += chunk.content

        print()  # Print a newline for formatting

        # Add AI response to the conversation
        messages.append(AIMessage(content=ai_response))

# Run the chatbot
if __name__ == "__main__":
    chatbot()
