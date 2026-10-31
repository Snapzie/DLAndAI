"""Run from the project root: uv run uvicorn App.main:app --reload."""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import sys
from threading import Lock
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from markdown_it import MarkdownIt
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError


ROOT = Path(__file__).resolve().parents[1]
COOKIE = "dog_chat_session"
logger = logging.getLogger(__name__)
markdown = MarkdownIt("commonmark", {"html": False}).enable("table").disable("image")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)


def create_app(agent_factory=None, database=None):
    @asynccontextmanager
    async def lifespan(app):
        if agent_factory is None:
            load_dotenv(ROOT / ".env")
            # Existing src modules use imports relative to the src directory.
            sys.path.insert(0, str(ROOT / "src"))
            from agent import build_agent
            from db import engine
            from App import database as database_queries

            app.state.agent = build_agent()
            app.state.database = database_queries
        else:
            app.state.agent = agent_factory()
            app.state.database = database
        app.state.sessions = {}
        try:
            yield
        finally:
            if agent_factory is None:
                engine.dispose()

    app = FastAPI(title="DogChat", lifespan=lifespan, docs_url=None, redoc_url=None)

    def new_session(request, response):
        session_id = str(uuid4())
        app.state.sessions[session_id] = Lock()
        response.set_cookie(COOKIE, session_id, httponly=True, samesite="strict", secure=request.url.scheme == "https")
        return session_id

    def reset_session(request, response, wait=False):
        session_id = request.cookies.get(COOKIE)
        lock = app.state.sessions.get(session_id)
        if lock is not None:
            if not lock.acquire(blocking=wait):
                raise HTTPException(409, "A reply is still being prepared. Please wait.")
            try:
                app.state.agent.checkpointer.delete_thread(session_id)
                app.state.sessions.pop(session_id, None)
            finally:
                lock.release()
        new_session(request, response)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        response = HTMLResponse((ROOT / "App" / "index.html").read_text(), headers={"Cache-Control": "no-store"})
        # Finish an active turn before deleting its checkpoints.
        reset_session(request, response, wait=True)
        return response

    @app.post("/api/conversation")
    def reset(request: Request, response: Response):
        reset_session(request, response)
        return {"ok": True}

    def read_database(query, response, *args):
        response.headers["Cache-Control"] = "no-store"
        try:
            return getattr(app.state.database, query)(*args)
        except SQLAlchemyError:
            logger.exception("Database browser request failed")
            raise HTTPException(503, "The database records could not be loaded. Please try again.") from None

    @app.get("/api/database/dogs")
    def list_dogs(response: Response):
        return read_database("list_dogs", response)

    @app.get("/api/database/dogs/{dog_id}")
    def dog_details(dog_id: int, response: Response):
        result = read_database("get_dog", response, dog_id)
        if result is None:
            raise HTTPException(404, "This dog record could not be found.")
        return result

    @app.get("/api/database/dogs/{dog_id}/shows")
    def dog_shows(dog_id: int, response: Response):
        return {"shows": read_database("list_shows", response, dog_id)}

    @app.get("/api/database/dogs/{dog_id}/litters")
    def dog_litters(dog_id: int, response: Response):
        return {"litters": read_database("list_litters", response, dog_id)}

    @app.get("/api/database/dogs/{dog_id}/shows/{show_id}/results")
    def dog_show_results(dog_id: int, show_id: int, response: Response):
        return {"results": read_database("get_results", response, dog_id, show_id)}

    @app.post("/api/chat")
    def chat(body: ChatRequest, request: Request):
        question = body.message.strip()
        if not question:
            raise HTTPException(422, "Please enter a question.")
        session_id = request.cookies.get(COOKIE)
        lock = app.state.sessions.get(session_id)
        if lock is None:
            raise HTTPException(409, "Your conversation expired. Start a new conversation.")
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "A reply is still being prepared. Please wait.")
        try:
            result = app.state.agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"configurable": {"thread_id": session_id}},
            )
            answer = result["messages"][-1].text
            if not answer.strip():
                raise ValueError("Agent returned no answer text")
            return {"text": answer, "html": markdown.render(answer)}
        except Exception:
            logger.exception("Chat request failed")
            raise HTTPException(503, "I couldn't complete that request. Please try again.") from None
        finally:
            lock.release()

    return app


app = create_app()
