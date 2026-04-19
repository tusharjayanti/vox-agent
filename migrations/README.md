# Migrations

Migrations are plain `.sql` files, numbered sequentially. They are never auto-generated — always hand-written and reviewed before being applied.

## Convention

```
migrations/
├── 001_init.sql          # conversations + turns tables
├── 002_evaluations.sql   # evaluations table
└── 003_pgvector.sql      # Phase 2: documents table + ivfflat index
```

- File names are `NNN_description.sql` with zero-padded three-digit numbers.
- Each file is idempotent where possible (use `CREATE TABLE IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`).
- Never edit a migration that has already been applied to any environment — add a new numbered file instead.

## Applying migrations

```bash
uv run python scripts/init_db.py
```

`init_db.py` reads `POSTGRES_DSN` from the environment, connects via asyncpg, and executes each `.sql` file in numeric order.
