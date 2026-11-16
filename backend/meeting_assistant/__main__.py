import uvicorn

uvicorn.run("meeting_assistant.main:app", host="0.0.0.0", port=8000, reload=False)
