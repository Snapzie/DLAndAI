import os
import pyodbc
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.pool import NullPool

load_dotenv()
pyodbc.pooling = False
connection_variable = os.environ.get("DATABASE_CONNECTION_ENV", "DATABASE_AGENT")

connection_url = URL.create(
    "mssql+pyodbc",
    query={"odbc_connect": os.environ[connection_variable]},
)

engine = create_engine(connection_url, poolclass=NullPool)
