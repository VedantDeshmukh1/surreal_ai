import os
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

def create_conversation_chain(api_key, model_name="llama3-8b-8192", memory_length=5):
    groq_chat = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name
    )

    system_prompt = 'You are an AI-powered virtual interviewer. Conduct the interview professionally and analyze the candidate\'s responses.'
    memory = ConversationBufferWindowMemory(k=memory_length, memory_key="chat_history", return_messages=True)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}")
    ])

    conversation = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )

    return conversation

def get_response(conversation, user_input):
    return conversation.predict(human_input=user_input)