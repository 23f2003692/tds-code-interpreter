from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import sys
from io import StringIO
import traceback
import os
import json

from google import genai
from google.genai import types

app = FastAPI()

# -------------------------
# CORS (IMPORTANT)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# -------------------------
# MODELS
# -------------------------
class CodeRequest(BaseModel):
    code: str

class CodeResponse(BaseModel):
    error: List[int]
    result: str


# -------------------------
# HEALTH ROUTE (VERY IMPORTANT)
# -------------------------
@app.get("/")
async def health():
    return {"status": "ok"}


# -------------------------
# TOOL FUNCTION
# -------------------------
def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code)
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}

    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}

    finally:
        sys.stdout = old_stdout


# -------------------------
# AI ERROR ANALYSIS
# -------------------------
def analyze_error_with_ai(code: str, tb: str) -> List[int]:

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return []

    client = genai.Client(api_key=api_key)

    prompt = f"""
Analyze this Python code and its error traceback.
Return ONLY the line number(s) where the error occurred.

CODE:
{code}

TRACEBACK:
{tb}
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "error_lines": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.INTEGER),
                    )
                },
                required=["error_lines"],
            ),
        ),
    )

    data = json.loads(response.text)
    return data["error_lines"]


# -------------------------
# MAIN ENDPOINT
# -------------------------
@app.post("/code-interpreter", response_model=CodeResponse)
async def code_interpreter(request: CodeRequest):

    execution = execute_python_code(request.code)

    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    else:
        error_lines = analyze_error_with_ai(
            request.code,
            execution["output"]
        )

        return {
            "error": error_lines,
            "result": execution["output"]
        }