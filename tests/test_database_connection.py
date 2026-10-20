import os
import unittest
from unittest.mock import patch

from src.database import connection


class DatabaseConnectionTests(unittest.TestCase):
    def test_local_postgres_settings_use_compose_host_port(self) -> None:
        environment = {
            "POSTGRES_USER": "test_user",
            "POSTGRES_PASSWORD": "test_password",
            "POSTGRES_DB": "test_database",
        }
        with patch.dict(os.environ, environment, clear=True), patch.object(
            connection.psycopg, "connect"
        ) as connect:
            connection.connect_to_database()

        connect.assert_called_once_with(
            host="localhost",
            port="5433",
            dbname="test_database",
            user="test_user",
            password="test_password",
        )

    def test_database_url_takes_precedence(self) -> None:
        with patch.dict(
            os.environ, {"DATABASE_URL": "postgresql://example"}, clear=True
        ), patch.object(connection.psycopg, "connect") as connect:
            connection.connect_to_database()

        connect.assert_called_once_with("postgresql://example")


if __name__ == "__main__":
    unittest.main()
