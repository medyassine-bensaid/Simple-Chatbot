import os
import time
from fastapi import FastAPI, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from langtrace_python_sdk import langtrace
from langfuse.openai import openai 

load_dotenv()


langtrace.init(
    api_key=os.getenv("LANGTRACE_API_KEY"),
    api_host=os.getenv("LANGTRACE_HOST")
)

REQUEST_COUNT = Counter("bot_requests_total", "Total chatbot requests")
REQUEST_LATENCY = Histogram("bot_request_latency_seconds", "Latency of chat requests")

SYSTEM_PROMPT = "You are a Senior Cloud & DevOps Assistant. Provide structured, production-ready answers."

app = FastAPI(title="Green-DevOps-Bot")

class Question(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"status": "DevOps Bot is online", "environment": "Docker"}

@app.post("/chat")
def chat(question: Question):
    REQUEST_COUNT.inc()
    start_time = time.time()

    # The Langfuse-wrapped OpenAI client automatically handles:
    # - Trace creation
    # - Generation logging
    # - Token usage (input/output)
    # - Latency timing
    completion = openai.chat.completions.create(
        name="devops-chat-request",
        model="llama-3.1-8b-instant",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question.message}
        ],
        metadata={
            "project": "AI-Monitoring-Lab",
            "agent": "FastAPI-Groq-Llama3"
        }
    )

    response_text = completion.choices[0].message.content

    # Observe latency for Prometheus
    REQUEST_LATENCY.observe(time.time() - start_time)

    return {"response": response_text}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health():
    return {"status": "ok"}