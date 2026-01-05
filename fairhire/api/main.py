from fastapi import FastAPI
from fairhire.api.routes import router

app = FastAPI(title="FairHire Auditor", version="0.1.0", description="AI Hiring Bias Detection API")
app.include_router(router, prefix="/api/v1")

@app.get("/health")
def health(): return {"status": "ok"}

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
