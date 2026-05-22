from typing import Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models.hosted import HostedAssistant
from app.models.oss import OSSAssistant

router = APIRouter()

sessions: Dict[str, Dict[str, object]] = {}


class ChatRequest(BaseModel):
	session_id: str
	prompt: str = ""


class ChatResponse(BaseModel):
	hosted_response: str
	oss_response: str


def get_or_create_session(session_id: str) -> Dict[str, object]:
	if session_id not in sessions:
		try:
			sessions[session_id] = {
				"hosted": HostedAssistant(),
				"oss": OSSAssistant(),
			}
		except ValueError as e:
			raise HTTPException(status_code=500, detail=str(e))
	return sessions[session_id]


@router.post(
	"/chat",
	response_model=ChatResponse,
	responses={500: {"description": "Session initialization failed"}},
)
async def chat_endpoint(request: ChatRequest):
	"""
	Sends the user prompt to both the Hosted and OSS assistants simultaneously
	and returns both responses while tracking short-term memory state.
	"""
	session = get_or_create_session(request.session_id)
	hosted_resp = session["hosted"].generate_response(request.prompt)
	oss_resp = session["oss"].generate_response(request.prompt)
	return ChatResponse(hosted_response=hosted_resp, oss_response=oss_resp)


@router.post("/clear")
async def clear_session(request: ChatRequest):
	"""Clears the short-term conversational history for a specific session."""
	if request.session_id in sessions:
		sessions[request.session_id]["hosted"].memory.clear()
		sessions[request.session_id]["oss"].memory.clear()
		return {"status": "success", "message": f"Memory cleared for session {request.session_id}"}
	return {"status": "ignored", "message": "Session not found"}
