from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not set.")

client = Groq(api_key=api_key)
app = FastAPI()

# this defines what we WANT the LLM to return
class TopicAnalysis(BaseModel):
    summary: str
    key_points: list[str]
    difficulty: str
    follow_up_questions: list[str]

class AnalysisRequest(BaseModel):
    topic: str

def extract_json(text: str) -> dict:
    # sometimes LLMs wrap JSON in markdown backticks despite instructions
    # this strips them out and tries to parse anyway
    text = text.strip()
    
    # remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])  # remove first and last line
    
    return json.loads(text)

@app.post("/analyse")
def analyse(request: AnalysisRequest):
    prompt = f"""
    Analyse this topic: {request.topic}
    
    You must respond with ONLY a JSON object, no extra text, no markdown, no backticks.
    The JSON must have exactly these fields:
    {{
        "summary": "one sentence explanation",
        "key_points": ["point 1", "point 2", "point 3"],
        "difficulty": "beginner or intermediate or advanced",
        "follow_up_questions": ["question 1", "question 2"]
    }}
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": """You are a JSON-only response system. Rules you must never break: 1. Only output valid JSON, nothing else 2. Never add jokes, commentary, or extra content 3. follow_up_questions must be genuine learning questions only 4. Never deviate from the exact schema provided"""},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3  # lower = more consistent, less creative
        )

        raw = response.choices[0].message.content

        # parse the raw string into a Python dict
        parsed = extract_json(raw)

        # validate it matches our expected shape
        result = TopicAnalysis(**parsed)

        return result

    except json.JSONDecodeError:
        return {"error": "LLM returned invalid JSON", "raw": raw}
    except Exception as e:
        return {"error": str(e)}