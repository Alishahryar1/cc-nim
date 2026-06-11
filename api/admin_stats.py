"""Persistent token usage statistics for admin dashboard."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from loguru import logger

DB_PATH = Path("/tmp/fcc_admin_stats.db")


class TokenStats:
    """Track token usage per model/provider with SQLite persistence."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or DB_PATH
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS token_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                provider_id TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_model ON token_usage(model)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_provider ON token_usage(provider_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON token_usage(timestamp)
        """)
        conn.commit()
        conn.close()

    def record_usage(
        self,
        model: str,
        provider_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record token usage for a request."""
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute(
                "INSERT INTO token_usage (model, provider_id, input_tokens, output_tokens) VALUES (?, ?, ?, ?)",
                (model, provider_id, input_tokens, output_tokens),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("Failed to record token stats: {}", e)

    def get_model_stats(self, hours: int = 24) -> list[dict[str, Any]]:
        """Return token stats per model for the last N hours."""
        try:
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.execute(
                """
                SELECT model,
                       SUM(input_tokens) as input_tokens,
                       SUM(output_tokens) as output_tokens,
                       COUNT(*) as requests
                FROM token_usage
                WHERE timestamp >= datetime('now', '-' || ? || ' hours')
                GROUP BY model
                ORDER BY (input_tokens + output_tokens) DESC
                """,
                (hours,),
            )
            result = [
                {
                    "model": row[0],
                    "input_tokens": row[1] or 0,
                    "output_tokens": row[2] or 0,
                    "total_tokens": (row[1] or 0) + (row[2] or 0),
                    "requests": row[3] or 0,
                }
                for row in cursor.fetchall()
            ]
            conn.close()
            return result
        except Exception as e:
            logger.warning("Failed to get model stats: {}", e)
            return []

    def get_provider_stats(self, hours: int = 24) -> list[dict[str, Any]]:
        """Return token stats per provider for the last N hours."""
        try:
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.execute(
                """
                SELECT provider_id,
                       SUM(input_tokens) as input_tokens,
                       SUM(output_tokens) as output_tokens,
                       COUNT(*) as requests
                FROM token_usage
                WHERE timestamp >= datetime('now', '-' || ? || ' hours')
                GROUP BY provider_id
                ORDER BY (input_tokens + output_tokens) DESC
                """,
                (hours,),
            )
            result = [
                {
                    "provider_id": row[0],
                    "input_tokens": row[1] or 0,
                    "output_tokens": row[2] or 0,
                    "total_tokens": (row[1] or 0) + (row[2] or 0),
                    "requests": row[3] or 0,
                }
                for row in cursor.fetchall()
            ]
            conn.close()
            return result
        except Exception as e:
            logger.warning("Failed to get provider stats: {}", e)
            return []

    def get_totals(self, hours: int = 24) -> dict[str, int]:
        """Return total stats for the last N hours."""
        try:
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.execute(
                """
                SELECT SUM(input_tokens) as input_tokens,
                       SUM(output_tokens) as output_tokens,
                       COUNT(*) as requests
                FROM token_usage
                WHERE timestamp >= datetime('now', '-' || ? || ' hours')
                """,
                (hours,),
            )
            row = cursor.fetchone()
            conn.close()
            return {
                "total_tokens": (row[0] or 0) + (row[1] or 0),
                "input_tokens": row[0] or 0,
                "output_tokens": row[1] or 0,
                "requests": row[2] or 0,
            }
        except Exception as e:
            logger.warning("Failed to get totals: {}", e)
            return {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0, "requests": 0}

    def reset(self) -> None:
        """Clear all statistics."""
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute("DELETE FROM token_usage")
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("Failed to reset stats: {}", e)

    def export_json(self, hours: int = 24) -> str:
        """Export stats as JSON."""
        import json

        return json.dumps(
            {
                "models": self.get_model_stats(hours),
                "providers": self.get_provider_stats(hours),
                "totals": self.get_totals(hours),
            },
            indent=2,
        )


# Global instance
token_stats = TokenStats()
