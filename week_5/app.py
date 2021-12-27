"""
Practice FastAPI
"""

# author: Seungyun Baek

from fastapi import FastAPI

app = FastAPI(title="MLOps Basics App")


@app.get("/")
async def home():
    """
    Home page
    """
    return "<h2>This is a sample NLP Project</h2>"
