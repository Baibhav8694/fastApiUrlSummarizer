from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Initialize the pipeline with the summarization task using the model of your choice
summarization_pipeline = pipeline("summarization")

class SummarizeRequest(BaseModel):
    url: str

class SummarizeResponse(BaseModel):
    summary: str

@app.post("/summarize")
async def summarize(request: SummarizeRequest) -> SummarizeResponse:
    try:
        # Call the Hugging Face summarization pipeline to summarize the URL
        summarized_text = summarization_pipeline(request.url, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        
        return SummarizeResponse(summary=summarized_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
