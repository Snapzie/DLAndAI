# DogChat

A minimal FastAPI frontend for the pedigree-dog LangChain agent, backed by SQL
Server. See the [app guide](<App - AI Generated/README.md>) for the interface and local run guide.

## Run locally

Install Microsoft ODBC Driver 18 for SQL Server, configure `.env`, then run:

```bash
uv sync
uv run uvicorn --app-dir "App - AI Generated" main:app --reload
```

Open <http://127.0.0.1:8000>. Local development defaults to `DATABASE_AGENT`.
`DATABASE_CONNECTION_ENV` selects a different connection environment variable
when explicitly set.
The new Docker image uses Python 3.12 without changing the local Python setting.

## Deploy to Render with Azure SQL

The Docker image serves both the page and API. No separate static site is needed.
It installs Microsoft ODBC Driver 18, uses `uv.lock`, and starts one Uvicorn worker
on Render's `PORT` (10000 by default). Database connections close after use:
both SQLAlchemy and pyodbc pooling are disabled to let idle Azure SQL compute pause.

1. Confirm that the Azure `DogChat` database contains your imported records and
   `dogchat_agent` has read-only access.
2. Commit the application, `Dockerfile`, `.dockerignore`, `.gitignore`,
   `pyproject.toml`, and `uv.lock` to a private Git repository. The ignore files
   exclude `.env`, local data, credentials, and common database exports. Ignore
   rules do not remove files previously committed; check that no secrets or data
   exports are already tracked before pushing.
3. Open the [Render dashboard](https://dashboard.render.com/), select
   **New → Web Service**, and connect your repository.
4. Choose these settings:

   | Setting | Value |
   | --- | --- |
   | Language | Docker |
   | Dockerfile path | `./Dockerfile` |
   | Root directory | `dog_chat` for the parent DLAndAI repository; leave empty for a standalone DogChat repository |
   | Instance type | Free |
   | Region | A nearby European region, such as Frankfurt if available |
   | Docker command | Leave empty; use the Dockerfile command |

5. Add these secrets in Render's **Environment** settings, without surrounding
   `.env` quotes:

   | Key | Value |
   | --- | --- |
   | `OPENAI_API_KEY` | Your OpenAI API key |
   | `DATABASE_AGENT_AZURE` | The Azure ODBC connection string below |
   | `DATABASE_CONNECTION_ENV` | `DATABASE_AGENT_AZURE` |

   ```text
   DRIVER={ODBC Driver 18 for SQL Server};SERVER=tcp:dogchat.database.windows.net,1433;DATABASE=DogChat;UID=dogchat_agent;PWD={YOUR_ACTUAL_PASSWORD};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=60;
   ```

   Enter these values directly in Render's Environment settings. Do not upload
   your local `.env` file or add the local `DATABASE_AGENT` value to Render.
   `DATABASE_CONNECTION_ENV` tells the application to read `DATABASE_AGENT_AZURE`.
   Do not use the administrator/import credentials for the application.
6. Open the Render service's **Connect → Outbound** panel. Add every listed IP
   range to the Azure SQL server firewall as its corresponding start/end address.
   Keep your existing client-IP rule for local administration.
7. Deploy the service. Check the build/runtime logs, then open its HTTPS
   `onrender.com` URL. An initial build or deployment can fail until secrets and
   firewall access are configured; redeploy after correcting them.
8. Verify that the database panel loads and that a chat question and follow-up
   work. Reload should clear chat memory. A paused Azure database may need time
   to resume; retry if the initial request times out.

Keep one worker and one instance while sessions and checkpoints are stored in
memory. The container trusts forwarded headers for Render's HTTPS proxy; use it
behind a trusted proxy when deploying elsewhere. The app currently has no sign-in
or rate limiting, so restrict access before sharing it beyond trusted users.

Render's Free service sleeps after inactivity, and memory resets on restart.
Keep Azure SQL overage billing disabled to pause at its monthly free limit.
OpenAI calls remain separately billed. Preparing these files does not deploy
the application or create cloud resources.

Official references:

- [Docker on Render](https://render.com/docs/docker)
- [Render web services](https://render.com/docs/web-services)
- [Render outbound IP ranges](https://render.com/docs/outbound-ip-addresses)
- [Render Free service limits](https://render.com/docs/free)
- [Azure SQL free tier](https://learn.microsoft.com/en-us/azure/azure-sql/database/free-tier?view=azuresql)
- [Microsoft ODBC installation](https://learn.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server?view=sql-server-ver17)

## Optional local Docker check

If Docker is installed:

```bash
docker build -t dogchat .
docker run --rm -p 10000:10000 --env-file /path/to/private-render.env dogchat
```

The separate private environment file needs `OPENAI_API_KEY`,
`DATABASE_AGENT_AZURE` (the Azure connection string), and
`DATABASE_CONNECTION_ENV=DATABASE_AGENT_AZURE`. Do not use the project's local
`.env` file for this check. Use Docker env-file syntax
(`KEY=value`, with no surrounding quotes). Keep the file outside the repository.
The Azure firewall must allow your computer's public IP. Open
<http://127.0.0.1:10000>.

## Tests

```bash
uv run python -B -m unittest discover -s "App - AI Generated" -p 'test_*.py'
```

The tests use fake agents and isolated SQLite records, without live OpenAI or
Azure SQL requests.
