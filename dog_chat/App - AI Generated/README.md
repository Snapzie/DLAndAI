# DogChat

A single HTML page and a FastAPI server running the existing agent. No frontend
build step, JavaScript packages, or external page assets are required.

## Run locally

From the project root:

```bash
uv sync
uv run uvicorn App.main:app --reload
```

Open <http://127.0.0.1:8000>.

The server uses the existing `.env` settings:

- `OPENAI_API_KEY`: the agent's OpenAI API key.
- `DATABASE_AGENT`: the SQL Server ODBC connection string used by `src/db.py`.
- `DATABASE_CONNECTION_ENV`: optional connection variable name; defaults to
  `DATABASE_AGENT`. On Render, set it to `DATABASE_AGENT_AZURE` and enter that
  Azure connection string directly in Render's Environment settings. Do not
  upload the local `.env` or local database credentials.

Install the SQL Server ODBC driver named in your connection string. Credentials
stay on the server. The page receives only answer text and rendered Markdown.

Each browser session has a separate conversation stored in memory. Follow-up
questions retain context. New conversation and page reloads both delete the
previous thread and start a new one. If a reply is still running, a page reload
waits for it to finish before clearing its memory. Tabs share the session cookie,
so reloading one tab also resets the conversation used by other tabs.
Restarting the server clears all conversations.

On desktop (900px and wider), the left-hand Database panel lists four selected
dogs alphabetically. This selection does not restrict the agent's searches.
Expand a dog for its recorded details and titles, then Shows
and an individual show for its class results, judges, grades, placements, and
awards. Sections load directly from the database when opened, independently of
the agent and chat memory. Missing records and loading failures are shown in
the panel, with a retry option. Reload the page to refresh loaded records.
The panel is hidden on smaller screens and makes no requests there.

Litters sits alongside Shows under each dog. Expand it to see breeding litters
where that dog is a recorded parent, newest first. Each litter has a plain
date-and-partner heading, recorded puppy count, and linked puppy names and birth
dates. Dividers separate litters; there are no further collapsibles inside.
Recorded puppy counts can differ from the number of linked individual records.

## Later deployment

For the Dockerfile and step-by-step Azure SQL + Render setup, see the
[deployment guide](../README.md#deploy-to-render-with-azure-sql).

Use a Python host or container that supports your SQL Server ODBC driver and can
reach the database. Set the required environment variables through the host's secret
settings, install the project dependencies, and run:

```bash
uv run uvicorn App.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Set the port to the value required by your host. Serve the page and API together
behind the host's HTTPS proxy. One worker is required because sessions and agent
checkpoints live in process memory. Multiple workers or replicas need shared
session storage and a persistent checkpointer first.

This prototype has no sign-in or rate limiting; anyone with access to its URL can
use the agent and its API budget. Put it behind your host's access control before
sharing beyond trusted users. Nothing is deployed by creating this folder.

## Check the web integration

```bash
uv run python -B -m unittest discover -s App -p 'test_*.py'
```

These checks use a fake agent and do not access OpenAI, SQL Server, or `.env`.
