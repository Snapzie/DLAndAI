import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

load_dotenv()

connection_url = URL.create(
    "mssql+pyodbc",
    query={"odbc_connect": os.environ["DATABASE_AGENT"]},
)

engine = create_engine(connection_url, pool_pre_ping=True)