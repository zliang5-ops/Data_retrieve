import gc
import getpass
import gzip
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd
import wrds


DEFAULT_BASE_DIR = Path.home() / "wrds_raw_downloads"
BASE_DIR = Path(os.environ.get("WRDS_RAW_BASE_DIR", str(DEFAULT_BASE_DIR))).expanduser()

DEFAULT_WRDS_MAX_WORKERS = 1
WRDS_CONNECT_RETRIES = int(os.environ.get("WRDS_CONNECT_RETRIES", "6"))
WRDS_CONNECT_RETRY_SECONDS = int(os.environ.get("WRDS_CONNECT_RETRY_SECONDS", "30"))
FINAL_GZIP_COMPRESSLEVEL = int(os.environ.get("WRDS_FINAL_GZIP_COMPRESSLEVEL", "5"))

QUOTE_TABLE_PREFIXES = ("ctq_", "cqm_", "cq_")
TRADE_TABLE_PREFIX = "ctm_"


def _safe_rollback(conn):
    try:
        if getattr(conn, "connection", None) is not None:
            conn.connection.rollback()
    except Exception:
        pass


def get_wrds_backend_context(conn) -> dict:
    sql = """
    select
        pg_backend_pid() as backend_pid,
        current_user as current_user,
        current_database() as current_database,
        current_setting('application_name', true) as application_name
    """
    try:
        df = conn.raw_sql(sql, chunksize=None)
        if df.empty:
            return {}
        row = df.iloc[0].to_dict()
        return {str(k): row.get(k) for k in row.keys()}
    except Exception:
        return {}


def try_get_wrds_query_id(conn) -> str | None:
    sql_candidates = [
        "select query_id from pg_stat_activity where pid = pg_backend_pid()",
        "select queryid as query_id from pg_stat_activity where pid = pg_backend_pid()",
    ]
    for sql in sql_candidates:
        try:
            df = conn.raw_sql(sql, chunksize=None)
            if df.empty:
                continue
            value = df.iloc[0].get("query_id")
            if pd.isna(value):
                continue
            return str(value)
        except Exception:
            continue
    return None


def format_wrds_exception(exc: Exception) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    orig = getattr(exc, "orig", None)
    if orig is not None:
        parts.append(f"orig={type(orig).__name__}: {orig}")
        pgcode = getattr(orig, "pgcode", None)
        if pgcode:
            parts.append(f"pgcode={pgcode}")
        diag = getattr(orig, "diag", None)
        if diag is not None:
            for attr in (
                "message_primary",
                "message_detail",
                "message_hint",
                "statement_position",
                "schema_name",
                "table_name",
                "column_name",
                "datatype_name",
                "constraint_name",
            ):
                value = getattr(diag, attr, None)
                if value:
                    parts.append(f"{attr}={value}")
    return " | ".join(parts)


def run_wrds_query(conn, sql: str, *, params=None, label: str, chunksize=500000):
    ctx = get_wrds_backend_context(conn)
    backend_pid = ctx.get("backend_pid")
    print(
        f"[WRDS QUERY START] label={label} | backend_pid={backend_pid} | params={params}",
        flush=True,
    )
    try:
        return conn.raw_sql(sql, params=params, chunksize=chunksize)
    except Exception as exc:
        err_text = format_wrds_exception(exc)
        _safe_rollback(conn)
        query_id = try_get_wrds_query_id(conn)
        print(
            f"[WRDS QUERY FAIL] label={label} | backend_pid={backend_pid} | "
            f"query_id={query_id or 'unavailable'} | error={err_text}",
            flush=True,
        )
        raise RuntimeError(
            f"WRDS query failed | label={label} | backend_pid={backend_pid} | "
            f"query_id={query_id or 'unavailable'} | error={err_text}"
        ) from exc


def list_wrds_tables(conn, library: str):
    ctx = get_wrds_backend_context(conn)
    backend_pid = ctx.get("backend_pid")
    print(f"[WRDS LIST TABLES] library={library} | backend_pid={backend_pid}", flush=True)
    try:
        return conn.list_tables(library=library)
    except Exception as exc:
        err_text = format_wrds_exception(exc)
        _safe_rollback(conn)
        query_id = try_get_wrds_query_id(conn)
        print(
            f"[WRDS LIST TABLES FAIL] library={library} | backend_pid={backend_pid} | "
            f"query_id={query_id or 'unavailable'} | error={err_text}",
            flush=True,
        )
        raise RuntimeError(
            f"WRDS list_tables failed | library={library} | backend_pid={backend_pid} | "
            f"query_id={query_id or 'unavailable'} | error={err_text}"
        ) from exc


def ensure_base_dir(base_dir: Path) -> Path:
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot create/write base directory: {base_dir}. "
            f"Set WRDS_RAW_BASE_DIR to a writable path."
        ) from exc
    return base_dir


def split_symbol(symbol: str):
    symbol = str(symbol).strip().upper()
    if "." in symbol:
        root, suffix = symbol.split(".", 1)
    else:
        root, suffix = symbol, ""
    return root, suffix


def get_wrds_credentials() -> tuple[str, str]:
    username = str(os.environ.get("WRDS_USERNAME", "")).strip()
    password = str(os.environ.get("WRDS_PASSWORD", "")).strip()

    if not username:
        default_username = getpass.getuser()
        username = input(f"Enter your WRDS username [{default_username}]: ").strip() or default_username
    if not password:
        password = getpass.getpass("Enter your WRDS password: ").strip()

    os.environ["WRDS_USERNAME"] = username
    os.environ["WRDS_PASSWORD"] = password
    return username, password


def is_wrds_connection_limit_error(exc: Exception):
    return "too many connections for role" in str(exc).lower()


def create_wrds_connection(username: str, password: str, load_library_list: bool = True):
    if not username or not password:
        raise ValueError("WRDS username and password are required before connecting.")

    os.environ["WRDS_USERNAME"] = username
    os.environ["WRDS_PASSWORD"] = password

    last_exc = None
    for attempt in range(1, WRDS_CONNECT_RETRIES + 1):
        conn = wrds.Connection(
            autoconnect=False,
            wrds_username=username,
            wrds_password=password,
        )
        try:
            make_conn = getattr(conn, "_Connection__make_sa_engine_conn")
            make_conn(raise_err=True)
            if conn.engine is None:
                raise ConnectionError(f"Failed to connect to WRDS as {username}")
            if load_library_list:
                conn.load_library_list()
            return conn
        except Exception as exc:
            last_exc = exc
            try:
                conn.close()
            except Exception:
                pass
            if not is_wrds_connection_limit_error(exc) or attempt == WRDS_CONNECT_RETRIES:
                raise
            wait_seconds = WRDS_CONNECT_RETRY_SECONDS * attempt
            print(
                f"WRDS connection limit reached for {username}; retrying in {wait_seconds}s "
                f"({attempt}/{WRDS_CONNECT_RETRIES})",
                flush=True,
            )
            time.sleep(wait_seconds)

    raise last_exc


def file_has_data(path: Path):
    return path.exists() and path.stat().st_size > 0


def asset_year_dir(symbol: str, year: int) -> Path:
    base_dir = ensure_base_dir(BASE_DIR)
    out = base_dir / symbol.upper() / str(year)
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_daily_table_index(conn):
    tables = list_wrds_tables(conn, library="taqmsec")
    trade_tables = {}
    quote_tables = {}

    for table_name in tables:
        name = table_name.lower()

        if name.startswith(TRADE_TABLE_PREFIX) and len(name) == len(TRADE_TABLE_PREFIX) + 8:
            try:
                table_date = pd.to_datetime(name[-8:], format="%Y%m%d").date()
            except Exception:
                continue
            trade_tables[table_date] = table_name
            continue

        quote_prefix = next((prefix for prefix in QUOTE_TABLE_PREFIXES if name.startswith(prefix)), None)
        if quote_prefix is None or len(name) != len(quote_prefix) + 8:
            continue

        try:
            table_date = pd.to_datetime(name[-8:], format="%Y%m%d").date()
        except Exception:
            continue
        quote_tables[table_date] = table_name

    out = []
    for table_date in sorted(set(trade_tables) | set(quote_tables)):
        out.append((table_date, trade_tables.get(table_date), quote_tables.get(table_date)))

    print(
        f"Indexed taqmsec daily tables | trade days: {len(trade_tables)} | quote days: {len(quote_tables)}",
        flush=True,
    )
    return out


def get_daily_tables_in_range(table_index, start_date: date, end_date: date):
    return [(d, trade_t, quote_t) for d, trade_t, quote_t in table_index if start_date <= d <= end_date]


def year_date_range(year: int):
    if year < 2026:
        return date(year, 1, 1), date(year, 12, 31)
    return date(2026, 1, 1), date(2026, 1, 1)


def write_gzip_csv_atomic(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    df.to_csv(tmp_path, index=False, compression="gzip")
    os.replace(tmp_path, path)


def fetch_raw_trades_one_day(conn, table_name: str, symbol: str) -> pd.DataFrame:
    sym_root, sym_suffix = split_symbol(symbol)

    sql = f"""
    select
        date,
        time_m,
        sym_root,
        sym_suffix,
        price,
        size
    from taqmsec.{table_name}
    where sym_root = %(sym_root)s
      and coalesce(sym_suffix, '') = %(sym_suffix)s
      and price > 0
      and size > 0
    order by date, time_m
    """

    df = run_wrds_query(
        conn,
        sql,
        label=f"raw_trades:{table_name}:{symbol.upper()}",
        params={
            "sym_root": sym_root,
            "sym_suffix": sym_suffix,
        },
        chunksize=None,
    )

    required_cols = {"date", "time_m", "sym_root", "sym_suffix", "price", "size"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {table_name}: {missing}")

    return df


def fetch_raw_quotes_one_day(conn, table_name: str, symbol: str) -> pd.DataFrame:
    sym_root, sym_suffix = split_symbol(symbol)

    sql = f"""
    select
        date,
        time_m,
        sym_root,
        sym_suffix,
        bid,
        bidsiz,
        ask,
        asksiz
    from taqmsec.{table_name}
    where sym_root = %(sym_root)s
      and coalesce(sym_suffix, '') = %(sym_suffix)s
      and bid > 0
      and ask > 0
      and ask >= bid
    order by date, time_m
    """

    df = run_wrds_query(
        conn,
        sql,
        label=f"raw_quotes:{table_name}:{symbol.upper()}",
        params={
            "sym_root": sym_root,
            "sym_suffix": sym_suffix,
        },
        chunksize=None,
    )

    required_cols = {"date", "time_m", "sym_root", "sym_suffix", "bid", "bidsiz", "ask", "asksiz"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {table_name}: {missing}")

    return df


def make_done_marker(path: Path, row_count: int):
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(f"rows={row_count}\nfinished_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    os.replace(tmp_path, path)


def copy_gzip_stream(src: Path, dst_fh):
    with src.open("rb") as src_fh:
        shutil.copyfileobj(src_fh, dst_fh)


def combine_daily_files(daily_files, final_path: Path):
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_final = final_path.with_name(final_path.name + ".tmp")
    with tmp_final.open("wb") as out_fh:
        for day_file in daily_files:
            if file_has_data(day_file):
                copy_gzip_stream(day_file, out_fh)
    os.replace(tmp_final, final_path)


def finalize_year_outputs(symbol: str, year: int, daily_tables, trade_daily_dir: Path, quote_daily_dir: Path, trade_file: Path, quote_file: Path):
    trade_daily_files = []
    quote_daily_files = []
    all_trade_done = True
    all_quote_done = True
    any_trade_rows = False
    any_quote_rows = False

    for trading_day, trade_table, quote_table in daily_tables:
        day_str = trading_day.strftime("%Y%m%d")
        if trade_table:
            trade_day_file = trade_daily_dir / f"{symbol.upper()}_{day_str}_raw_consolidate_trade.csv.gz"
            trade_done = trade_day_file.with_suffix(trade_day_file.suffix + ".done")
            if trade_done.exists():
                trade_daily_files.append(trade_day_file)
                any_trade_rows = any_trade_rows or file_has_data(trade_day_file)
            else:
                all_trade_done = False
        if quote_table:
            quote_day_file = quote_daily_dir / f"{symbol.upper()}_{day_str}_raw_consolidate_quote.csv.gz"
            quote_done = quote_day_file.with_suffix(quote_day_file.suffix + ".done")
            if quote_done.exists():
                quote_daily_files.append(quote_day_file)
                any_quote_rows = any_quote_rows or file_has_data(quote_day_file)
            else:
                all_quote_done = False

    if all_trade_done:
        if any_trade_rows:
            combine_daily_files(trade_daily_files, trade_file)
        else:
            trade_file.touch()

    if all_quote_done:
        if any_quote_rows:
            combine_daily_files(quote_daily_files, quote_file)
        else:
            quote_file.touch()

    return all_trade_done, all_quote_done


def save_one_day_raw(conn, symbol: str, trading_day, trade_table, quote_table, trade_daily_dir: Path, quote_daily_dir: Path):
    day_str = trading_day.strftime("%Y%m%d")

    if trade_table:
        trade_day_file = trade_daily_dir / f"{symbol.upper()}_{day_str}_raw_consolidate_trade.csv.gz"
        trade_done = trade_day_file.with_suffix(trade_day_file.suffix + ".done")
        if trade_done.exists() and trade_day_file.exists():
            print(f"Skipping finished trade day {symbol.upper()} {day_str}", flush=True)
        else:
            print(f"Querying raw trades for {symbol.upper()} on {day_str} from {trade_table}", flush=True)
            trades_df = fetch_raw_trades_one_day(conn, trade_table, symbol)
            write_gzip_csv_atomic(trades_df, trade_day_file)
            make_done_marker(trade_done, len(trades_df))
            print(f"Finished raw trades for {symbol.upper()} on {day_str} | rows: {len(trades_df)}", flush=True)
            del trades_df
            gc.collect()

    if quote_table:
        quote_day_file = quote_daily_dir / f"{symbol.upper()}_{day_str}_raw_consolidate_quote.csv.gz"
        quote_done = quote_day_file.with_suffix(quote_day_file.suffix + ".done")
        if quote_done.exists() and quote_day_file.exists():
            print(f"Skipping finished quote day {symbol.upper()} {day_str}", flush=True)
        else:
            print(f"Querying raw quotes for {symbol.upper()} on {day_str} from {quote_table}", flush=True)
            quotes_df = fetch_raw_quotes_one_day(conn, quote_table, symbol)
            write_gzip_csv_atomic(quotes_df, quote_day_file)
            make_done_marker(quote_done, len(quotes_df))
            print(f"Finished raw quotes for {symbol.upper()} on {day_str} | rows: {len(quotes_df)}", flush=True)
            del quotes_df
            gc.collect()


def save_one_year_raw(conn, table_index, symbol: str, year: int):
    ydir = asset_year_dir(symbol, year)

    trade_dir = ydir / "raw_consolidate_trade"
    quote_dir = ydir / "raw_consolidate_quote"
    trade_daily_dir = trade_dir / "daily"
    quote_daily_dir = quote_dir / "daily"
    trade_dir.mkdir(parents=True, exist_ok=True)
    quote_dir.mkdir(parents=True, exist_ok=True)
    trade_daily_dir.mkdir(parents=True, exist_ok=True)
    quote_daily_dir.mkdir(parents=True, exist_ok=True)

    trade_file = trade_dir / f"{symbol.upper()}_{year}_raw_consolidate_trade.csv.gz"
    quote_file = quote_dir / f"{symbol.upper()}_{year}_raw_consolidate_quote.csv.gz"

    start_date, end_date = year_date_range(year)
    daily_tables = get_daily_tables_in_range(table_index, start_date, end_date)

    if not daily_tables:
        print(f"No taqmsec daily tables found for {symbol.upper()} {year}", flush=True)
        trade_file.touch()
        quote_file.touch()
        return

    trade_complete, quote_complete = finalize_year_outputs(
        symbol,
        year,
        daily_tables,
        trade_daily_dir,
        quote_daily_dir,
        trade_file,
        quote_file,
    )
    if trade_complete and quote_complete and trade_file.exists() and quote_file.exists():
        print(f"Skipping existing completed year for {symbol.upper()} {year}", flush=True)
        return

    print(
        f"Now downloading {symbol.upper()} | year folder: {year} | date range: {start_date} to {end_date}",
        flush=True,
    )

    for trading_day, trade_table, quote_table in daily_tables:
        save_one_day_raw(conn, symbol, trading_day, trade_table, quote_table, trade_daily_dir, quote_daily_dir)

    trade_complete, quote_complete = finalize_year_outputs(
        symbol,
        year,
        daily_tables,
        trade_daily_dir,
        quote_daily_dir,
        trade_file,
        quote_file,
    )

    if trade_complete:
        print(f"Saved trade file: {trade_file}", flush=True)
    else:
        print(f"Trade file incomplete for {symbol.upper()} {year}; rerun will resume.", flush=True)

    if quote_complete:
        print(f"Saved quote file: {quote_file}", flush=True)
    else:
        print(f"Quote file incomplete for {symbol.upper()} {year}; rerun will resume.", flush=True)

    print(f"Finished {symbol.upper()} {year}", flush=True)


def run_one_symbol_year(symbol: str, year: int, table_index, username: str, password: str):
    conn = create_wrds_connection(username, password, load_library_list=False)
    try:
        save_one_year_raw(conn, table_index, symbol, year)
        return symbol, year, "ok", ""
    except Exception as exc:
        return symbol, year, "error", str(exc)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def main():
    ensure_base_dir(BASE_DIR)
    print(f"WRDS raw download base dir: {BASE_DIR}", flush=True)

    assets = [
        "AAPL",
        "AMZN",
        "AVGO",
        "BRKA",
        "GOOG",
        "META",
        "MSFT",
        "TSLA",
        "WMT",
        "NVDA",
    ]

    start_year = 2016
    end_year = 2026

    username, password = get_wrds_credentials()
    conn = create_wrds_connection(username, password)

    try:
        table_index = build_daily_table_index(conn)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    tasks = [(symbol, year) for symbol in assets for year in range(start_year, end_year + 1)]
    max_workers = 1

    for symbol, year in tasks:
        sym, yr, status, message = run_one_symbol_year(symbol, year, table_index, username, password)
        if status == "ok":
            print(f"[DONE] {sym} {yr}", flush=True)
        else:
            print(f"[FAIL] {sym} {yr} | {message}", flush=True)

    print("All tasks finished.", flush=True)


if __name__ == "__main__":
    main()
