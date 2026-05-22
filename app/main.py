from fastapi import FastAPI

from app.api.routes import router as api_router

import uvicorn

app = FastAPI(
	title="Ollive AI Dual Assistant Engine",
	description="Backend powering simultaneous side-by-side Frontier and Open Source AI models.",
	version="1.0.0",
)

app.include_router(api_router, prefix="/api")


@app.get("/")
def read_root():
	return {"status": "online", "message": "FastAPI AI Engine running successfully."}


if __name__ == "__main__":
	uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
