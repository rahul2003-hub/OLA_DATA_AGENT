from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

def pick_llm(level: str):
    """
    Picks the appropriate LLM based on the level of the question.

    Args:
        level (str): The level of the question, can be "low", "medium", or "high".

    Returns:
        ChatGroq: The LLM instance to be used.
    """
    if level.lower() == "low":
        llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)
    elif level.lower() == "medium":
        llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0)
    elif level.lower() == "high":
        llm = ChatGroq(model_name="qwen/qwen3.8-27b", temperature=0)
    else:
        raise ValueError(f"Unsupported level: {level}")

    return llm

if __name__ == "__main__":
    llm_obj = pick_llm("medium")  
    print(llm_obj.invoke("What is the capital of France?"))