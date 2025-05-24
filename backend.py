
from dotenv import load_dotenv
load_dotenv()

#Step1: Setup Pydantic Model (Schema Validation)
from pydantic import BaseModel
from typing import List


class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool


#Step2: Setup AI Agent from FrontEnd Request
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ai_agent import get_response_from_ai_agent

ALLOWED_MODEL_NAMES=["llama3-70b-8192", "mixtral-8x7b-32768", "llama-3.3-70b-versatile", "gpt-4o-mini"]

app=FastAPI(title="mitrr AI Agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
def chat_endpoint(request: RequestState): 
    """
    API Endpoint to interact with the Chatbot using LangGraph and search tools.
    It dynamically selects the model specified in the request
    """
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": "Invalid model name. Kindly select a valid AI model"}
    
    llm_id = request.model_name
    query = request.messages
    allow_search = request.allow_search
    system_prompt = request.system_prompt
    provider = request.model_provider

    # Create AI Agent and get response from it! 
    response=get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider)
    return response

# Summarization endpoint
class SummarizeRequest(BaseModel):
    text: str
    model_name: str = "llama-3.3-70b-versatile"
    system_prompt: str = "You are a helpful AI assistant that summarizes text concisely."

@app.post("/summarize")
async def summarize_text(request: SummarizeRequest):
    """
    Summarize the given text using the specified model.
    """
    try:
        # Use the same agent logic but with summarization prompt
        response = get_response_from_ai_agent(
            llm_id=request.model_name,
            query=[request.text],
            allow_search=False,  # Don't search for summarization
            system_prompt=request.system_prompt,
            provider="Groq" if "llama" in request.model_name or "mixtral" in request.model_name.lower() else "OpenAI"
        )
        return {"summary": response}
    except Exception as e:
        return {"error": f"Error during summarization: {str(e)}"}

#Step3: Run app & Explore Swagger UI Docs
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
