"""Exercise the web boundary without credentials or external services."""

from contextlib import ExitStack
from datetime import date
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from langchain.messages import AIMessage

from App.main import COOKIE, create_app


class WebTests(unittest.TestCase):
    def setUp(self):
        self.contexts = ExitStack()
        self.addCleanup(self.contexts.close)
        self.agent = SimpleNamespace(invoke=Mock(), checkpointer=SimpleNamespace(delete_thread=Mock()))
        self.agent.invoke.return_value = {"messages": [AIMessage(content=[
            {"type": "reasoning", "encrypted_content": "internal-reasoning"},
            {"type": "text", "text": "**A recorded answer**"},
        ])]}
        self.database = SimpleNamespace(
            list_dogs=Mock(return_value={"dogs": [{"dog_id": 1, "pedigree_name": "Example"}], "total_count": 42, "show_count": 12}),
            get_dog=Mock(return_value={"dog_id": 1, "pedigree_name": "Example", "birth_date": date(2020, 1, 1), "titles": []}),
            list_shows=Mock(return_value=[{"show_id": 2, "date": date(2024, 1, 1), "location": "Herning", "organizer": None}]),
            list_litters=Mock(return_value=[{"birth_date": date(2021, 1, 1), "partner_name": "Partner", "puppy_count": 2, "puppies": []}]),
            get_results=Mock(return_value=[{"show_class": "Open Class", "awards": ["CK", "CERT"]}]),
        )
        self.app = create_app(lambda: self.agent, self.database)
        self.client = self.contexts.enter_context(TestClient(self.app))
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_page_and_cookie(self):
        page = self.client.get("/")
        self.assertIn("Every dog has a story", page.text)
        self.assertIn('aria-label="Conversation"', page.text)
        self.assertEqual(page.headers["cache-control"], "no-store")
        self.assertTrue(self.client.cookies.get(COOKIE))

    def test_follow_up_uses_same_thread_and_returns_only_answer(self):
        thread = self.client.cookies.get(COOKIE)
        for question in ("Find a dog", "Who are its parents?"):
            response = self.client.post("/api/chat", json={"message": question})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["text"], "**A recorded answer**")
            self.assertIn("<strong>", response.json()["html"])
            self.assertNotIn("internal-reasoning", response.text)
            self.assertEqual(self.agent.invoke.call_args.kwargs["config"]["configurable"]["thread_id"], thread)

    def test_separate_browsers_have_separate_threads(self):
        other = self.contexts.enter_context(TestClient(self.app))
        other.get("/")
        self.assertNotEqual(self.client.cookies.get(COOKIE), other.cookies.get(COOKIE))
        other.post("/api/chat", json={"message": "Hello"})
        self.assertEqual(self.agent.invoke.call_args.kwargs["config"]["configurable"]["thread_id"], other.cookies.get(COOKIE))

    def test_new_conversation_clears_old_checkpoint(self):
        previous = self.client.cookies.get(COOKIE)
        response = self.client.post("/api/conversation")
        self.assertEqual(response.status_code, 200)
        self.assertNotEqual(self.client.cookies.get(COOKIE), previous)
        self.agent.checkpointer.delete_thread.assert_called_once_with(previous)
        self.assertNotIn(previous, self.app.state.sessions)

    def test_reload_clears_old_checkpoint_and_starts_new_thread(self):
        previous = self.client.cookies.get(COOKIE)
        self.client.post("/api/chat", json={"message": "Remember this dog"})
        self.assertEqual(self.client.get("/").status_code, 200)
        current = self.client.cookies.get(COOKIE)
        self.assertNotEqual(current, previous)
        self.agent.checkpointer.delete_thread.assert_called_once_with(previous)
        self.assertNotIn(previous, self.app.state.sessions)
        self.client.post("/api/chat", json={"message": "Hello"})
        self.assertEqual(self.agent.invoke.call_args.kwargs["config"]["configurable"]["thread_id"], current)

    def test_invalid_input_and_expired_session_do_not_call_agent(self):
        for message in ("", "   ", "x" * 8001):
            self.assertEqual(self.client.post("/api/chat", json={"message": message}).status_code, 422)
        self.client.cookies.clear()
        response = self.client.post("/api/chat", json={"message": "Hello"})
        self.assertEqual(response.status_code, 409)
        self.agent.invoke.assert_not_called()

    def test_concurrent_turn_is_rejected(self):
        lock = self.app.state.sessions[self.client.cookies.get(COOKIE)]
        with lock:
            self.assertEqual(self.client.post("/api/chat", json={"message": "Hello"}).status_code, 409)
            self.assertEqual(self.client.post("/api/conversation").status_code, 409)
        self.agent.invoke.assert_not_called()

    def test_failure_hides_details_and_allows_next_turn(self):
        self.agent.invoke.side_effect = RuntimeError("private database details")
        with patch("App.main.logger"):
            response = self.client.post("/api/chat", json={"message": "Hello"})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private database details", response.text)
        self.agent.invoke.side_effect = None
        self.assertEqual(self.client.post("/api/chat", json={"message": "Try again"}).status_code, 200)

    def test_markdown_rejects_html_and_unsafe_links(self):
        text = '<script>alert(1)</script>\n\n[click](javascript:alert(1))\n\n| Dog | Title |\n| --- | --- |\n| Example | CERT |'
        self.agent.invoke.return_value = {"messages": [AIMessage(content=text)]}
        rendered = self.client.post("/api/chat", json={"message": "Hello"}).json()["html"]
        self.assertNotIn("<script>", rendered)
        self.assertNotIn('href="javascript:', rendered)
        self.assertIn("<table>", rendered)

    def test_database_routes_work_independently_of_agent_and_chat(self):
        thread = self.client.cookies.get(COOKIE)
        response = self.client.get("/api/database/dogs")
        self.assertEqual(response.json()["dogs"][0]["pedigree_name"], "Example")
        self.assertEqual(response.json()["total_count"], 42)
        self.assertEqual(response.json()["show_count"], 12)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(self.client.get("/api/database/dogs/1").json()["birth_date"], "2020-01-01")
        self.assertEqual(self.client.get("/api/database/dogs/1/shows").json()["shows"][0]["show_id"], 2)
        self.assertEqual(self.client.get("/api/database/dogs/1/shows/2/results").json()["results"][0]["awards"], ["CK", "CERT"])
        self.database.get_results.assert_called_once_with(1, 2)
        litters = self.client.get("/api/database/dogs/1/litters")
        self.assertEqual(litters.json()["litters"][0]["birth_date"], "2021-01-01")
        self.database.list_litters.assert_called_once_with(1)
        self.assertEqual(self.client.cookies.get(COOKIE), thread)
        self.agent.invoke.assert_not_called()
        self.agent.checkpointer.delete_thread.assert_not_called()

    def test_database_missing_dog(self):
        self.database.get_dog.return_value = None
        self.assertEqual(self.client.get("/api/database/dogs/999").status_code, 404)

    def test_database_failure_hides_details(self):
        from sqlalchemy.exc import OperationalError

        self.database.list_dogs.side_effect = OperationalError("private query", {}, Exception("private connection"))
        with patch("App.main.logger"):
            response = self.client.get("/api/database/dogs")
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private", response.text)


if __name__ == "__main__":
    unittest.main()
