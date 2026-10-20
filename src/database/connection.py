import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / "docker" / ".env")


def connect_to_database() -> psycopg.Connection:
    """Connect to PostgreSQL using DATABASE_URL or the project's POSTGRES_* settings."""
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        return psycopg.connect(database_url)

    required_variables = ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")
    missing_variables = [
        variable for variable in required_variables if not os.environ.get(variable)
    ]
    if missing_variables:
        missing = ", ".join(missing_variables)
        raise RuntimeError(
            f"Missing required database environment variables: {missing}"
        )

    return psycopg.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=os.environ.get("DB_PORT", "5433"),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )
