"""
Idempotent migration runner.

Reads .sql files from migrations/ in numeric order. For each file,
checks the schema_migrations table — if not already applied, runs it
in a transaction and records the application. Safe to run on every
deploy.

Usage:
    uv run python scripts/init_db.py
"""
import asyncio
import sys
from pathlib import Path

import asyncpg

from voxagent.config import settings


MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"


async def run_migrations() -> None:
    if not MIGRATIONS_DIR.is_dir():
        print(f"No migrations directory at {MIGRATIONS_DIR}", file=sys.stderr)
        sys.exit(1)

    conn = await asyncpg.connect(settings.postgres_dsn)
    try:
        # Bootstrap schema_migrations so we can query it
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                filename TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        applied = {
            row["filename"]
            for row in await conn.fetch(
                "SELECT filename FROM schema_migrations"
            )
        }

        sql_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        if not sql_files:
            print("No migration files found.")
            return

        applied_count = 0
        for sql_file in sql_files:
            if sql_file.name in applied:
                print(f"[skip]    {sql_file.name} (already applied)")
                continue

            print(f"[apply]   {sql_file.name}")
            sql = sql_file.read_text()
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO schema_migrations (filename) VALUES ($1) "
                    "ON CONFLICT (filename) DO NOTHING",
                    sql_file.name,
                )
            applied_count += 1

        print(f"\nMigrations complete. {applied_count} new, "
              f"{len(applied)} already applied.")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(run_migrations())
