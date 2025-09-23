# Pytest Guide

## Introduction

This guide explains the structure and expectations for running tests with pytest in this repository. Pytest can also execute legacy unittest suites. Examples below assume pytest as the default.
The necessary dependencies install when you run `pip install -r requirements-dev.txt` or bootstrap the dev container.

There are two primary testing styles in Python:

1. **unittest** – built into the standard library, mostly kept for backward compatibility.
2. **pytest** – concise, expressive, and the tooling we rely on for new tests.

We default to pytest for all new test code.

## Generic Flow

Pytest walks through discovery and fixture setup in a predictable order:

1. **Session start**
   - Load `pytest.ini`
   - Discover eligible test files
   - Import `conftest.py` files (global first, then per-directory)
2. **Session fixtures**
   - Run setup for fixtures with `scope="session"`
3. **Per-module cycle**
   - For each test file:
     - Run `scope="module"` fixture setup
     - For each test class:
       - Run `scope="class"` fixture setup
       - For each test function:
         - Run `scope="function"` fixture setup
         - Execute the test body
         - Tear down function fixtures
       - Tear down class fixtures
     - Tear down module fixtures
4. **Session teardown**
   - Tear down remaining session fixtures

## Key Points to Keep in Mind

- **Central config**: Keep `pytest.ini` at the repo root so pytest picks it up automatically.
- **Global fixtures**: Store session-wide fixtures in `tests/conftest.py` for effortless reuse.
- **Scoped fixtures**: Add a `conftest.py` to any test subfolder when you need fixtures confined to that directory.
- **Automatic discovery**: Never import `conftest.py` manually—pytest discovers it for you.
- **Discovery names**: Use `*_test.py` for files and `test_*` for test functions or methods.
- **Fixture lifecycle**: Code before `yield` runs during setup; code after `yield` handles teardown for that scope.
- **Test types**:
  - **Unit** – pure logic, orchestration, and error handling; rely on mocks/fakes for fast feedback.
  - **Integration** – SQL correctness, migrations, and “does this really read/write?” checks; use real dependencies sparingly for higher confidence.

## `conftest.py` Template

Use this template as a starting point for repo-wide fixtures. Trim what you do not need and rename fixtures to match the language of your test suite.

```python
"""
CONFTEST TEMPLATE (pytest)
-------------------------

Put this file in:
  - tests/conftest.py                    # global fixtures for all tests
  - tests/<subsuite>/conftest.py         # fixtures only for that folder

WHAT THIS GIVES YOU:
- Session/module/function-scoped fixtures (with clear examples)
- Real-DB integration flow (session connection, per-test truncate+seed, safe skip)
- Unit-test helpers (MagicMock factories, monkeypatch helpers)
- Logging capture defaults (caplog)
- Utility fixtures: row fetcher, id/payload factories, tmp paths
- Optional CLI flag: --run-integration to explicitly include integration tests

NOTES:
- Fixture **names are NOT standardized**. Rename for readability in your codebase.
- Only test discovery names are strict (test_*.py, test_* functions).
- DB placeholders: using mysql-connector → **use %s**, not ?.
"""

import json
import logging
import os
import random
import string
from contextlib import closing
from typing import Callable, Iterable, Optional

import pytest
from unittest.mock import MagicMock

# -----------------------------------------------------------------------------
# OPTIONAL: Toggle integration tests with a CLI flag
#   - By default, we SKIP tests marked @pytest.mark.integration
#   - Run them by: pytest -q --run-integration
# -----------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run tests marked as integration (real DB / external deps).",
    )

def pytest_configure(config):
    # Register common custom markers so pytest doesn't warn
    config.addinivalue_line("markers", "unit: pure unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: integration tests (real DB)")
    config.addinivalue_line("markers", "slow: long-running tests")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return  # user explicitly asked to run integration tests
    skip_integration = pytest.mark.skip(reason="use --run-integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)

# =============================================================================
#                           SESSION-LEVEL CONFIG / ENV
# =============================================================================

@pytest.fixture(scope="session")
def app_env() -> dict:
    """
    Session-scoped app/test configuration.
    Reads env vars; provide safe defaults for local dev.
    Accessible to all tests; you can extend this shape anytime.
    """
    return {
        "OCAPI_HOST": os.getenv("OCAPI_HOST", "127.0.0.1"),
        "OCAPI_PORT": int(os.getenv("OCAPI_PORT", "3306")),
        "OCAPI_USER": os.getenv("OCAPI_USER", "root"),
        "OCAPI_PASS": os.getenv("OCAPI_PASS", "password"),
        "OCAPI_SCHEMA": os.getenv("OCAPI_SCHEMA", "test_schema"),
        # Optional: toggle autocommit
        "OCAPI_AUTOCOMMIT": os.getenv("OCAPI_AUTOCOMMIT", "false").lower() == "true",
    }

# =============================================================================
#                     SESSION-LEVEL DB CONNECTION (INTEGRATION)
# =============================================================================

@pytest.fixture(scope="session")
def conn(app_env):
    """
    One MySQL connection for the WHOLE test run (integration).
    - Yields a live connection if DB is reachable.
    - Yields None if not available (tests using it can skip gracefully).
    - DO NOT rely on __del__ of wrappers; explicitly close here.
    """
    # Lazy import so unit tests (without DB deps) don't import mysql libs
    try:
        from database import OcapiPrimaryDatabase  # your wrapper
    except Exception:
        # If your test suite sometimes runs outside the app context
        OcapiPrimaryDatabase = None  # type: ignore

    connection = None
    if OcapiPrimaryDatabase is not None:
        try:
            db = OcapiPrimaryDatabase(autocommit=app_env["OCAPI_AUTOCOMMIT"])
            connection = db.get_connection()
            # keep long sessions alive in big suites
            try:
                connection.ping(reconnect=True, attempts=1, delay=0)
            except Exception:
                pass
        except Exception as e:
            # DB not reachable; integration tests will skip when they see conn is None
            logging.getLogger(__name__).warning(f"DB connection unavailable: {e}")

    yield connection

    # teardown: close if open
    if connection is not None:
        try:
            connection.close()
        except Exception:
            pass

# =============================================================================
#                   PER-TEST DB ISOLATION (TRUNCATE + SEED)
# =============================================================================

@pytest.fixture()
def db_isolation(conn):
    """
    Function-scoped fixture to isolate DB state per test.
    - Truncates tables and seeds minimal rows BEFORE the test
    - Truncates again AFTER the test
    - Requires a real connection; will skip test if conn is None

    HOW TO USE:
        def test_something(conn, db_isolation):
            ...

    RENAME:
        You can rename 'db_isolation' to 'clean_and_seed' or anything you like.
    """
    if conn is None:
        pytest.skip("DB connection not available; skipping DB-dependent test")

    # ---- SETUP (before test) ----
    truncate_tables(conn, tables=["batch_reservations"])
    seed_batch_reservations(conn, rows=[
        (101, 1234, 0, {"from": "seed"}),
        (102, 1234, 0, {"from": "seed"}),
        (103, 1234, 0, {"from": "seed"}),
    ])
    if not getattr(conn, "autocommit", False):
        conn.commit()

    yield  # ---- run the test ----

    # ---- TEARDOWN (after test) ----
    truncate_tables(conn, tables=["batch_reservations"])
    if not getattr(conn, "autocommit", False):
        conn.commit()

def truncate_tables(conn, tables: Iterable[str]):
    """Utility to TRUNCATE a list of tables."""
    for t in tables:
        with closing(conn.cursor()) as cur:
            cur.execute(f"TRUNCATE TABLE {t}")

def seed_batch_reservations(conn, rows: list[tuple[int, int, int, dict]]):
    """
    Seed 'batch_reservations' rows.
    Each row: (id, company_id, status, payload_dict)
    """
    for (rid, company_id, status, payload) in rows:
        with closing(conn.cursor()) as cur:
            cur.execute(
                "INSERT INTO batch_reservations (id, company_id, status, payload) VALUES (%s,%s,%s,%s)",
                (rid, company_id, status, json.dumps(payload)),
            )

# =============================================================================
#                             LOGGING & OUTPUT DEFAULTS
# =============================================================================

@pytest.fixture(autouse=False)
def set_log_level(caplog):
    """
    Optional autouse fixture to set default logging level for every test.
    Set autouse=True if you want it ON by default in this scope.
    """
    caplog.set_level(logging.INFO)
    yield

# =============================================================================
#                         GENERIC UNIT-TEST UTIL FIXTURES
# =============================================================================

@pytest.fixture()
def fake_client() -> MagicMock:
    """
    A generic MagicMock 'client' you can hand to services.
    Extend per test as needed:
        fake_client.create.return_value = {...}
        fake_client.update.side_effect = RuntimeError("boom")
    """
    m = MagicMock(name="fake_client")
    m.create.return_value = {"id": 42}
    m.update.return_value = True
    return m

@pytest.fixture()
def make_mock() -> Callable[..., MagicMock]:
    """
    Factory fixture to build named MagicMocks easily:

        mm = make_mock("foo", spec=SomeClass)
        mm.method.return_value = 1
    """
    def _factory(name: Optional[str] = None, **kwargs) -> MagicMock:
        return MagicMock(name=name, **kwargs)
    return _factory

@pytest.fixture()
def id_factory() -> Callable[[int], list[int]]:
    """
    Make predictable ID lists for parametrized tests:

        ids = id_factory(3)  -> [1001, 1002, 1003]
    """
    def _make(n: int) -> list[int]:
        start = 1000
        return [start + i + 1 for i in range(n)]
    return _make

@pytest.fixture()
def payload_factory() -> Callable[[int], dict]:
    """
    Generate lightweight JSON-ish payloads:

        payload_factory(3) -> {'k': 'v-3', 'token': 'AB12...'}
    """
    def _make(seed: int = 0) -> dict:
        rnd = random.Random(seed)
        token = "".join(rnd.choice(string.ascii_uppercase + string.digits) for _ in range(8))
        return {"k": f"v-{seed}", "token": token}
    return _make

# =============================================================================
#                       SMALL DB HELPER (FUNCTION-AS-FIXTURE)
# =============================================================================

@pytest.fixture()
def fetch_row(conn):
    """
    Returns a callable: fetch_row(table, id) -> dict | None
    Usage:
        row = fetch_row("batch_reservations", 101)
    """
    if conn is None:
        pytest.skip("DB connection not available; skipping DB-dependent test")

    def _fetch(table: str, rid: int):
        with closing(conn.cursor(dictionary=True, buffered=True)) as cur:
            cur.execute(f"SELECT * FROM {table} WHERE id=%s", (rid,))
            return cur.fetchone()
    return _fetch

```

## Example Usage

The snippets below show the template fixtures in action.

### Unit test example (no DB)

Keep unit tests isolated by mocking external collaborators.

```python
# tests/test_service.py
def test_service_uses_client(fake_client, payload_factory):
    class Service:
        def __init__(self, client):
            self.client = client

        def create(self, payload):
            return self.client.create(payload)["id"]

    svc = Service(fake_client)
    pid = svc.create(payload_factory(7))
    fake_client.create.assert_called_once()
    assert pid == 42

```


### Integration test example (real DB)

Real database fixtures should be opt-in via `@pytest.mark.integration` and the `--run-integration` flag.

```python
# tests/test_mapper.py
import pytest
from contextlib import closing

@pytest.mark.integration
def test_updates_status(conn, db_isolation, fetch_row):
    before = fetch_row("batch_reservations", 101)
    assert before["status"] == 0

    with closing(conn.cursor()) as cur:
        cur.execute("UPDATE batch_reservations SET status=1 WHERE id IN (%s)", (101,))
    conn.commit()

    after = fetch_row("batch_reservations", 101)
    assert after["status"] == 1

```


## Tests Template

Copy the sections you need for new suites and adjust names to match the domain.

```python
"""
A SINGLE, COMPREHENSIVE PYTEST TEMPLATE FILE

HOW TO USE:
- Keep this file as a reference you can copy from.
- Take only the sections you need (DB fixtures, mocks, parametrized tests, etc.).
- Rename fixtures to anything you find readable. Fixture names are NOT strict.
- Keep test file names as test_*.py or *_test.py (pytest requirement).
"""

import json
import time
import logging
import pytest
from contextlib import closing
from unittest.mock import MagicMock, patch

# -----------------------------------------------------------------------------
# OPTIONAL: IMPORT YOUR REAL CODE UNDER TEST
# -----------------------------------------------------------------------------
# Example mapper (integration style)
# from integration.primary.mapper.batch_reservations_primary_mapper import BatchReservationsPrimaryMapper

# Example service (unit style)
# from services.my_service import MyService


# =============================================================================
#                            GLOBAL MARKS & CONVENTIONS
# =============================================================================
# You can define your own marks (labels). Then run with: pytest -m integration
pytestmark = []  # you can leave empty or set defaults per file

# Example of custom marks you may use across your repo:
# @pytest.mark.unit
# @pytest.mark.integration
# @pytest.mark.slow


# =============================================================================
#                            FIXTURE BASICS (CHEATSHEET)
# =============================================================================
# - Fixtures are reusable setup/teardown blocks.
# - Scope options: "function", "class", "module", "session".
# - Names are free-form; pick readable names. (Not a standard)
# - Use `autouse=True` for fixtures that should run automatically for ALL tests in
#   scope without being referenced as parameters.
# - Teardown code is placed AFTER the `yield`.

# ------------------------------
# Session-scoped example (runs once per test run)
# ------------------------------
@pytest.fixture(scope="session")
def app_config():
    """
    Example of a session-level config. Build once, reuse everywhere.
    Replace with your real config helper if needed.
    """
    return {
        "OCAPI": {
            "host": "localhost",
            "port": 3306,
            "user": "root",
            "password": "password",
            "schema": "test_schema",
        }
    }


# ------------------------------
# Module-scoped DB connection (integration)
# ------------------------------
@pytest.fixture(scope="module")
def conn(app_config):
    """
    Real MySQL connection used for INTEGRATION tests.

    IMPORTANT:
    - Keep autocommit=False so tests control visibility with explicit commit()
    - If you don't need DB in some tests, don't request this fixture
    - Name 'conn' is arbitrary; use anything readable
    """
    try:
        # Example using your DB wrapper (uncomment and adjust to your project)
        # from database import OcapiPrimaryDatabase
        # db = OcapiPrimaryDatabase(autocommit=False)
        # c = db.get_connection()

        # If you don’t have the DB locally, you can skip at runtime:
        # pytest.skip("DB not available; skipping integration tests")

        c = None  # <-- replace with your real connection as above
        if c is not None:
            try:
                # Keep idle connections from timing out in long runs
                c.ping(reconnect=True, attempts=1, delay=0)
            except Exception:
                pass
        yield c
    finally:
        if c is not None:
            try:
                c.close()
            except Exception:
                pass


# ------------------------------
# Function-scoped DB cleanup/seed (integration)
# Runs before and after EACH test function that requests 'conn'
# ------------------------------
@pytest.fixture()
def clean_and_seed(conn):
    """
    TRUNCATE + SEED before each test that needs DB, then TRUNCATE after.
    This keeps tests isolated and deterministic.

    NOTE:
    - This runs ONLY if the test function includes 'clean_and_seed' in its params
      OR if you set autouse=True.
    - 'clean_and_seed' is NOT a standard name; use any you like.
    """
    if conn is None:
        # If DB not available, skip the particular test
        pytest.skip("DB connection not available; skipping DB-dependent test")

    with closing(conn.cursor()) as cur:
        cur.execute("TRUNCATE TABLE batch_reservations")

    with closing(conn.cursor()) as cur:
        cur.execute(
            "INSERT INTO batch_reservations (id, company_id, status, payload) VALUES (%s,%s,%s,%s)",
            (101, 1234, 0, json.dumps({"from": "seed"})),
        )
        cur.execute(
            "INSERT INTO batch_reservations (id, company_id, status, payload) VALUES (%s,%s,%s,%s)",
            (102, 1234, 0, json.dumps({"from": "seed"})),
        )
    conn.commit()

    yield  # ---- tests run here ----

    with closing(conn.cursor()) as cur:
        cur.execute("TRUNCATE TABLE batch_reservations")
    conn.commit()


# ------------------------------
# Function-scoped pure unit fixture (no DB)
# ------------------------------
@pytest.fixture()
def fake_client():
    """
    Example of a fake/mocked dependency for unit tests.
    Return any object your code under test expects.
    """
    client = MagicMock()
    client.create.return_value = {"id": 42}
    client.update.return_value = True
    return client


# ------------------------------
# Autouse per-function fixture (applies to every test in this file)
# Use sparingly—nice for global logging tweaks, env vars, etc.
# ------------------------------
@pytest.fixture(autouse=False)
def _auto_logging_tweak(caplog):
    """
    Example 'autouse' (disabled by default).
    If you set autouse=True, this runs for EVERY test.
    """
    caplog.set_level(logging.INFO)
    yield
    # optional teardown here


# =============================================================================
#                          HELPER UTILITIES (OPTIONAL)
# =============================================================================
def fetch_reservation(conn, rid: int):
    """Helper for integration tests to fetch one row as dict."""
    with closing(conn.cursor(dictionary=True, buffered=True)) as cur:
        cur.execute(
            "SELECT id, status, created, modified FROM batch_reservations WHERE id=%s",
            (rid,),
        )
        return cur.fetchone()


# =============================================================================
#                                EXAMPLE TESTS
# =============================================================================
# ------------------------------
# 1) INTEGRATION-STYLE TESTS (REAL DB)
# ------------------------------
@pytest.mark.integration
def test_mapper_completed_updates_status_and_modified(conn, clean_and_seed):
    """
    Example integration test that uses real DB.
    - Uses 'conn' (module fixture) and 'clean_and_seed' (function fixture)
    """
    if conn is None:
        pytest.skip("DB not available")

    before = fetch_reservation(conn, 101)
    assert before["status"] == 0

    # Imagine calling your real mapper here:
    # affected = BatchReservationsPrimaryMapper.completed(conn, [101])
    # For demonstration without the real call:
    with closing(conn.cursor()) as cur:
        cur.execute(
            "UPDATE batch_reservations SET status=1 WHERE id IN (%s)",
            (101,),
        )
        affected = cur.rowcount
    conn.commit()

    after = fetch_reservation(conn, 101)
    assert affected == 1
    assert after["status"] == 1
    assert after["modified"] >= before["modified"]


@pytest.mark.integration
@pytest.mark.parametrize("ids,expected", [([101, 102], 2), ([999999], 0), ([], 0)])
def test_mapper_completed_various_inputs(conn, clean_and_seed, ids, expected):
    if conn is None:
        pytest.skip("DB not available")

    # Replace with a call to your mapper:
    affected = 0
    if ids:
        placeholders = ",".join(["%s"] * len(ids))
        with closing(conn.cursor()) as cur:
            cur.execute(
                f"UPDATE batch_reservations SET status=1 WHERE id IN ({placeholders})",
                tuple(ids),
            )
            affected = cur.rowcount
        conn.commit()

    assert affected == expected


# ------------------------------
# 2) UNIT-STYLE TESTS (NO DB) — MOCKS / FAKES
# ------------------------------
@pytest.mark.unit
def test_service_calls_client_once_and_returns_id(fake_client):
    """
    Example pure unit test for a 'service' that depends on a client.
    No DB; just mock/fake collaborators.
    """
    # Example service implementation shape:
    class MyService:
        def __init__(self, client):
            self.client = client
        def create_item(self, payload):
            res = self.client.create(payload)
            return res["id"]

    svc = MyService(fake_client)
    item_id = svc.create_item({"x": 1})
    fake_client.create.assert_called_once_with({"x": 1})
    assert item_id == 42


@pytest.mark.unit
def test_patch_function_in_imported_module_with_monkeypatch(monkeypatch):
    """
    Demonstrates monkeypatching a free function in a module under test.
    """
    # Suppose module_under_test.py has: def external_create(p): ...
    # and your code calls module_under_test.external_create(payload)
    class ModuleUnderTest:
        @staticmethod
        def external_create(payload):
            raise RuntimeError("Real network call not allowed in unit tests!")
        @staticmethod
        def create_wrapper(payload):
            return ModuleUnderTest.external_create(payload)

    def fake_create(payload):
        return {"ok": True, "id": 7}

    # monkeypatch the attribute on the module/class you call
    monkeypatch.setattr(ModuleUnderTest, "external_create", fake_create)

    assert ModuleUnderTest.create_wrapper({"a": 1}) == {"ok": True, "id": 7}


# ------------------------------
# 3) CAPTURE OUTPUT / LOGS / TEMP FILES
# ------------------------------
def test_capturing_stdout_and_logs(capsys, caplog):
    logging.getLogger(__name__).info("hello log")
    print("hello stdout")

    captured = capsys.readouterr()
    assert "hello stdout" in captured.out

    assert any("hello log" in rec.message for rec in caplog.records)


def test_tmp_path_and_files(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps({"x": 1}))
    assert p.read_text() == '{"x": 1}'


# ------------------------------
# 4) SKIP / XFAIL / TIMEOUT / SLOW EXAMPLES
# ------------------------------
@pytest.mark.skip(reason="Demonstration of skipping a test")
def test_skip_example():
    assert True  # will not run


@pytest.mark.xfail(reason="Known bug; update when fixed")
def test_xfail_example():
    assert 1 == 2  # marked xfail, doesn't fail the suite


@pytest.mark.slow
def test_slow_example():
    time.sleep(0.1)
    assert True

```
