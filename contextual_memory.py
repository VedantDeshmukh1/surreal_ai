import os
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

def generate_system_prompt(role):
    return f"""You are an AI assistant specialized in the role of {role}.Stick to the {role} role and do not deviate from it dont be imaginative. Your task is to provide accurate and helpful information related to this role, answering questions to the best of your capabilities. Always stay in character and provide responses that are consistent with your expertise in {role}. If a question falls outside your area of expertise, politely explain that it's beyond your specialized knowledge, but try to offer relevant information or suggest where the user might find an answer."""

def create_conversation_chain(api_key, role=None):
    llm = ChatGroq(temperature=0.7, api_key=api_key)
    memory = ConversationBufferMemory(return_messages=True)
    
    system_prompt = generate_system_prompt(role) if role else ""
    print(system_prompt)
    if system_prompt:
        memory.chat_memory.add_message(SystemMessage(content=system_prompt))
    
    template = """
    {history}
    Human: {human_input}
    AI: """
    
    prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
    
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True
    )
    return conversation

def get_response(conversation, user_input):
    return conversation.predict(human_input=user_input)