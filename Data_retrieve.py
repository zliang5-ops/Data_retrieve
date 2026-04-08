import gc
import getpass
import os
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import date
import pandas as pd
import wrds


DEFAULT_MAX_WORKERS = min(2, max(1, (os.cpu_count() or 1)))
DEFAULT_WRDS_MAX_WORKERS = max(1, int(os.environ.get("WRDS_MAX_WORKERS", str(DEFAULT_MAX_WORKERS))))
WRDS_CONNECT_RETRIES = int(os.environ.get("WRDS_CONNECT_RETRIES", "6"))
WRDS_CONNECT_RETRY_SECONDS = int(os.environ.get("WRDS_CONNECT_RETRY_SECONDS", "30"))


def split_symbol(symbol: str):
    symbol = symbol.strip().upper()
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
        username = input(f"Enter your WRDS username [{default_username}]:").strip() or default_username
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


def is_leap_year(year: int):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def year_bimonths(year: int):
    feb_end = 29 if is_leap_year(year) else 28
    return [
        (date(year, 1, 1), date(year, 2, feb_end), "B1"),
        (date(year, 3, 1), date(year, 4, 30), "B2"),
        (date(year, 5, 1), date(year, 6, 30), "B3"),
        (date(year, 7, 1), date(year, 8, 31), "B4"),
        (date(year, 9, 1), date(year, 10, 31), "B5"),
        (date(year, 11, 1), date(year, 12, 31), "B6"),
    ]


def build_minute_bars_with_sessions(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()

    df = raw_df.loc[:, ["date", "time_m", "price", "size"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    valid_mask = df["date"].notna() & (df["price"] > 0) & (df["size"] > 0)
    if not valid_mask.all():
        df = df.loc[valid_mask]

    if df.empty:
        return pd.DataFrame()

    time_str = df["time_m"].astype(str).str.strip()
    trade_td = pd.to_timedelta(time_str, errors="coerce")
    td_mask = trade_td.notna()
    if not td_mask.all():
        df = df.loc[td_mask]
        trade_td = trade_td.loc[td_mask]

    if df.empty:
        return pd.DataFrame()

    df["trade_dt"] = df["date"] + trade_td
    df["minute_dt"] = df["trade_dt"].dt.floor("min")

    hhmm = df["trade_dt"].dt.hour * 100 + df["trade_dt"].dt.minute

    df["session"] = "discard"
    df.loc[(hhmm >= 400) & (hhmm < 930), "session"] = "premarket"
    df.loc[(hhmm >= 930) & (hhmm < 1600), "session"] = "regular"
    df.loc[(hhmm >= 1600) & (hhmm < 2000), "session"] = "postmarket"
    session_mask = df["session"] != "discard"
    if not session_mask.all():
        df = df.loc[session_mask]

    if df.empty:
        return pd.DataFrame()

    df["trade_value"] = df["price"] * df["size"]
    df = df.sort_values(["minute_dt", "trade_dt"], kind="mergesort")

    g = df.groupby(["minute_dt", "session"], sort=True)

    bars = g.agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("size", "sum"),
        trade_value=("trade_value", "sum"),
    ).reset_index()

    bars["vwap"] = bars["trade_value"] / bars["volume"]

    vol_by_price = (
        df.groupby(["minute_dt", "session", "price"], sort=True)["size"]
        .sum()
        .rename("vol_at_price")
        .reset_index()
    )

    px_at_highest_volume = (
        vol_by_price.sort_values(
            ["minute_dt", "session", "vol_at_price", "price"],
            ascending=[True, True, False, False],
        )
        .drop_duplicates(subset=["minute_dt", "session"], keep="first")
        .rename(columns={"price": "px_at_highest_volume"})
        [["minute_dt", "session", "px_at_highest_volume"]]
        .reset_index(drop=True)
    )

    out = bars.merge(px_at_highest_volume, on=["minute_dt", "session"], how="left", sort=False, copy=False)

    out["date"] = out["minute_dt"].dt.date
    out["minute"] = out["minute_dt"].dt.time

    out = out[
        [
            "minute_dt",
            "date",
            "minute",
            "session",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "px_at_highest_volume",
        ]
    ].sort_values(["minute_dt", "session"], kind="mergesort").reset_index(drop=True)

    return out


def build_ctm_daily_table_index(conn):
    tables = conn.list_tables(library="taqmsec")
    out = []

    for t in tables:
        m = re.fullmatch(r"ctm_(\d{8})", t.lower())
        if m:
            d = pd.to_datetime(m.group(1), format="%Y%m%d").date()
            out.append((d, t))

    out.sort(key=lambda x: x[0])
    return out


def get_ctm_daily_tables_in_range(ctm_table_index, start_date, end_date):
    return [(d, t) for d, t in ctm_table_index if start_date <= d <= end_date]


def fetch_raw_ticks_one_day(conn, table_name: str, symbol: str) -> pd.DataFrame:
    sym_root, sym_suffix = split_symbol(symbol)

    sql = f"""
    select
        date,
        time_m,
        price,
        size
    from taqmsec.{table_name}
    where sym_root = %(sym_root)s
      and coalesce(sym_suffix, '') = %(sym_suffix)s
      and price > 0
      and size > 0
    """

    df = conn.raw_sql(
        sql,
        params={
            "sym_root": sym_root,
            "sym_suffix": sym_suffix,
        },
    )

    required_cols = {"date", "time_m", "price", "size"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {table_name}: {missing}")

    return df


def save_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)


def append_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not file_has_data(path)
    df.to_csv(path, index=False, mode="a", header=write_header)


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "minute_dt" in df.columns:
        df["minute_dt"] = pd.to_datetime(df["minute_dt"], errors="coerce")
    return df


def file_has_data(path: Path):
    return path.exists() and path.stat().st_size > 0


def required_files_exist(paths):
    return all(file_has_data(Path(path)) for path in paths)


def year_output_files(asset_dir: Path, symbol: str, year: int):
    year_all_file = asset_dir / f"{symbol}_{year}_1min_all_sessions.csv"
    year_regular_file = asset_dir / f"{symbol}_{year}_1min_regular.csv"
    if not required_files_exist([year_all_file, year_regular_file]):
        return {}

    files = {
        "all_sessions": year_all_file,
        "regular": year_regular_file,
    }
    for sess in ["premarket", "postmarket"]:
        f = asset_dir / f"{symbol}_{year}_1min_{sess}.csv"
        if file_has_data(f):
            files[sess] = f
    return files


def final_output_files(asset_dir: Path, symbol: str, start_year: int, end_year: int):
    final_all_file = asset_dir / f"{symbol}_{start_year}_{end_year}_1min_all_sessions.csv"
    final_regular_file = asset_dir / f"{symbol}_{start_year}_{end_year}_1min_regular.csv"
    if not required_files_exist([final_all_file, final_regular_file]):
        return {}

    files = {
        "all_sessions": final_all_file,
        "regular": final_regular_file,
    }
    for sess in ["premarket", "postmarket"]:
        f = asset_dir / f"{symbol}_{start_year}_{end_year}_1min_{sess}.csv"
        if file_has_data(f):
            files[sess] = f
    return files


def data_output_dirs(base_output_dir: Path, symbol: str):
    out = []
    seen = set()
    for data_dir in [base_output_dir, *sorted(base_output_dir.parent.glob("data_*_*"))]:
        if not re.fullmatch(r"data_\d{4}_\d{4}", data_dir.name):
            continue
        asset_dir = data_dir / symbol
        if asset_dir in seen or not asset_dir.exists():
            continue
        seen.add(asset_dir)
        out.append(asset_dir)
    return out


def find_existing_year_files(symbol: str, year: int, base_output_dir: Path):
    target_asset_dir = base_output_dir / symbol
    files = year_output_files(target_asset_dir, symbol, year)
    if files:
        return files

    for asset_dir in data_output_dirs(base_output_dir, symbol):
        if asset_dir == target_asset_dir:
            continue
        files = year_output_files(asset_dir, symbol, year)
        if files:
            target_asset_dir.mkdir(parents=True, exist_ok=True)
            copied = {}
            for kind, src in files.items():
                dest = target_asset_dir / src.name
                if not file_has_data(dest):
                    shutil.copy2(src, dest)
                copied[kind] = dest
            print(f"Reusing existing year {year} for {symbol} from {asset_dir}", flush=True)
            return copied

    return {}


def find_covering_final_files(symbol: str, year: int, base_output_dir: Path):
    candidates = []
    for asset_dir in data_output_dirs(base_output_dir, symbol):
        for all_file in asset_dir.glob(f"{symbol}_*_1min_all_sessions.csv"):
            m = re.fullmatch(rf"{re.escape(symbol)}_(\d{{4}})_(\d{{4}})_1min_all_sessions\.csv", all_file.name)
            if not m:
                continue

            start_year = int(m.group(1))
            end_year = int(m.group(2))
            if not (start_year <= year <= end_year):
                continue

            files = final_output_files(asset_dir, symbol, start_year, end_year)
            if files:
                candidates.append((end_year - start_year, start_year, end_year, files))

    if not candidates:
        return {}

    candidates.sort()
    return candidates[0][3]


def filter_csv_to_year(source_path: Path, dest_path: Path, year: int):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_name(dest_path.name + ".partial")
    temp_path.unlink(missing_ok=True)

    wrote_any = False
    for chunk in pd.read_csv(source_path, chunksize=250_000):
        if "date" in chunk.columns:
            years = pd.to_datetime(chunk["date"], errors="coerce").dt.year
        elif "minute_dt" in chunk.columns:
            years = pd.to_datetime(chunk["minute_dt"], errors="coerce").dt.year
        else:
            raise ValueError(f"Cannot filter {source_path} by year: missing date/minute_dt column")

        part = chunk.loc[years == year]
        if part.empty:
            continue

        part.to_csv(temp_path, index=False, mode="a", header=not wrote_any)
        wrote_any = True

    if wrote_any:
        temp_path.replace(dest_path)
        return True

    temp_path.unlink(missing_ok=True)
    dest_path.unlink(missing_ok=True)
    return False


def ensure_year_available_from_existing(symbol: str, year: int, base_output_dir: Path):
    files = find_existing_year_files(symbol, year, base_output_dir)
    if files:
        return files

    source_files = find_covering_final_files(symbol, year, base_output_dir)
    if not source_files:
        return {}

    target_asset_dir = base_output_dir / symbol
    target_asset_dir.mkdir(parents=True, exist_ok=True)

    extracted = {}
    for kind, source_path in source_files.items():
        dest = target_asset_dir / f"{symbol}_{year}_1min_{kind}.csv"
        if file_has_data(dest) or filter_csv_to_year(source_path, dest, year):
            extracted[kind] = dest

    if required_files_exist(
        [
            target_asset_dir / f"{symbol}_{year}_1min_all_sessions.csv",
            target_asset_dir / f"{symbol}_{year}_1min_regular.csv",
        ]
    ):
        print(f"Extracted existing year {year} for {symbol} from multi-year output", flush=True)
        return year_output_files(target_asset_dir, symbol, year)

    return {}


def save_session_split_files(df: pd.DataFrame, base_stem: Path):
    out = {}

    all_file = base_stem.with_name(base_stem.name + "_all_sessions.csv")
    save_csv(df, all_file)
    out["all_sessions"] = all_file

    for sess in ["premarket", "regular", "postmarket"]:
        part = df.loc[df["session"] == sess]
        if not part.empty:
            f = base_stem.with_name(base_stem.name + f"_{sess}.csv")
            save_csv(part, f)
            out[sess] = f

    return out


def combine_files(file_list, final_path: Path, sort_cols=None):
    if not file_list:
        return False

    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.with_name(final_path.name + ".partial")

    with temp_path.open("w", newline="", encoding="utf-8") as dst:
        wrote_any = False
        for src_path in file_list:
            src_path = Path(src_path)
            if not file_has_data(src_path):
                continue

            with src_path.open("r", newline="", encoding="utf-8") as src:
                header = src.readline()
                if not header:
                    continue

                if not wrote_any:
                    dst.write(header)
                    shutil.copyfileobj(src, dst)
                    wrote_any = True
                else:
                    shutil.copyfileobj(src, dst)

    if not wrote_any:
        temp_path.unlink(missing_ok=True)
        return False

    temp_path.replace(final_path)
    return True


def process_one_bimonth(
    conn,
    ctm_table_index,
    symbol: str,
    year: int,
    start_date: date,
    end_date: date,
    block_label: str,
    asset_dir: Path,
    print_state: dict,
):
    print(f"Processing {symbol} {year} {block_label}: {start_date} to {end_date}")

    daily_tables = get_ctm_daily_tables_in_range(ctm_table_index, start_date, end_date)

    if not daily_tables:
        print(f"No daily tables found for {symbol} {year} {block_label}")
        return {}

    base_stem = asset_dir / f"{symbol}_{year}_{block_label}_1min"
    block_all_file = base_stem.with_name(base_stem.name + "_all_sessions.csv")
    block_regular_file = base_stem.with_name(base_stem.name + "_regular.csv")
    block_premarket_file = base_stem.with_name(base_stem.name + "_premarket.csv")
    block_postmarket_file = base_stem.with_name(base_stem.name + "_postmarket.csv")

    if required_files_exist([block_all_file, block_regular_file]):
        print(f"Skipping completed block {symbol} {year} {block_label}", flush=True)
        files = {
            "all_sessions": block_all_file,
            "regular": block_regular_file,
        }
        if file_has_data(block_premarket_file):
            files["premarket"] = block_premarket_file
        if file_has_data(block_postmarket_file):
            files["postmarket"] = block_postmarket_file
        return files

    for stale in [block_all_file, block_regular_file, block_premarket_file, block_postmarket_file]:
        stale.unlink(missing_ok=True)

    wrote_any = False

    for _, table_name in daily_tables:
        raw_df = fetch_raw_ticks_one_day(conn, table_name, symbol)

        if raw_df.empty:
            del raw_df
            gc.collect()
            continue

        if not print_state["raw_printed"]:
            print("\nRaw data sample:")
            print(raw_df[["date", "time_m", "price", "size"]].head(5))
            print_state["raw_printed"] = True

        minute_df_day = build_minute_bars_with_sessions(raw_df)

        del raw_df
        gc.collect()

        if minute_df_day.empty:
            del minute_df_day
            gc.collect()
            continue

        if not print_state["processed_printed"]:
            print("\nProcessed data sample:")
            print(minute_df_day.head(5))
            print_state["processed_printed"] = True

        append_csv(minute_df_day, block_all_file)

        pre_df = minute_df_day.loc[minute_df_day["session"] == "premarket"]
        if not pre_df.empty:
            append_csv(pre_df, block_premarket_file)

        reg_df = minute_df_day.loc[minute_df_day["session"] == "regular"]
        if not reg_df.empty:
            append_csv(reg_df, block_regular_file)

        post_df = minute_df_day.loc[minute_df_day["session"] == "postmarket"]
        if not post_df.empty:
            append_csv(post_df, block_postmarket_file)

        wrote_any = True

        del minute_df_day
        gc.collect()

    if not wrote_any:
        print(f"No minute bars for {symbol} {year} {block_label}")
        return {}

    files = {"all_sessions": block_all_file}
    if file_has_data(block_premarket_file):
        files["premarket"] = block_premarket_file
    if file_has_data(block_regular_file):
        files["regular"] = block_regular_file
    if file_has_data(block_postmarket_file):
        files["postmarket"] = block_postmarket_file
    return files


def process_one_year(conn, ctm_table_index, symbol: str, year: int, asset_dir: Path, print_state: dict):
    year_regular_file = asset_dir / f"{symbol}_{year}_1min_regular.csv"
    year_all_file = asset_dir / f"{symbol}_{year}_1min_all_sessions.csv"
    year_premarket_file = asset_dir / f"{symbol}_{year}_1min_premarket.csv"
    year_postmarket_file = asset_dir / f"{symbol}_{year}_1min_postmarket.csv"

    if required_files_exist([year_all_file, year_regular_file]):
        print(f"Skipping completed year {year} for {symbol}", flush=True)
        out = {
            "all_sessions": year_all_file,
            "regular": year_regular_file,
        }
        if file_has_data(year_premarket_file):
            out["premarket"] = year_premarket_file
        if file_has_data(year_postmarket_file):
            out["postmarket"] = year_postmarket_file
        return out

    block_outputs = []

    for start_date, end_date, block_label in year_bimonths(year):
        files = process_one_bimonth(
            conn,
            ctm_table_index,
            symbol,
            year,
            start_date,
            end_date,
            block_label,
            asset_dir,
            print_state,
        )
        if files:
            block_outputs.append(files)
        gc.collect()

    if not block_outputs:
        print(f"No yearly output for {symbol} {year}")
        return {}

    year_files = {}

    all_block_files = [x["all_sessions"] for x in block_outputs if "all_sessions" in x]
    if combine_files(all_block_files, year_all_file, ["minute_dt", "session"]):
        year_files["all_sessions"] = year_all_file

    for sess in ["premarket", "regular", "postmarket"]:
        sess_block_files = [x[sess] for x in block_outputs if sess in x]
        if sess_block_files:
            year_sess_file = asset_dir / f"{symbol}_{year}_1min_{sess}.csv"
            if combine_files(sess_block_files, year_sess_file, ["minute_dt", "session"]):
                year_files[sess] = year_sess_file

    for files in block_outputs:
        for f in files.values():
            f.unlink(missing_ok=True)

    del block_outputs
    gc.collect()

    print(f"Finished year {year} for {symbol}")
    return year_files


def process_one_asset(conn, ctm_table_index, symbol: str, start_year: int, end_year: int, base_output_dir: Path, print_state: dict):
    asset_dir = base_output_dir / symbol
    asset_dir.mkdir(parents=True, exist_ok=True)
    final_regular_file = asset_dir / f"{symbol}_{start_year}_{end_year}_1min_regular.csv"
    final_all_file = asset_dir / f"{symbol}_{start_year}_{end_year}_1min_all_sessions.csv"

    if required_files_exist([final_all_file, final_regular_file]):
        print(f"Skipping completed asset {symbol}", flush=True)
        out = {
            "all_sessions": final_all_file,
            "regular": final_regular_file,
        }
        final_premarket_file = asset_dir / f"{symbol}_{start_year}_{end_year}_1min_premarket.csv"
        final_postmarket_file = asset_dir / f"{symbol}_{start_year}_{end_year}_1min_postmarket.csv"
        if file_has_data(final_premarket_file):
            out["premarket"] = final_premarket_file
        if file_has_data(final_postmarket_file):
            out["postmarket"] = final_postmarket_file
        return out

    yearly_outputs = []

    for year in range(start_year, end_year + 1):
        ensure_year_available_from_existing(symbol, year, base_output_dir)
        files = process_one_year(conn, ctm_table_index, symbol, year, asset_dir, print_state)
        if files:
            yearly_outputs.append(files)
        gc.collect()

    if not yearly_outputs:
        print(f"No output created for {symbol}")
        return {}

    asset_all_files = [x["all_sessions"] for x in yearly_outputs if "all_sessions" in x]
    combine_files(asset_all_files, final_all_file, ["minute_dt", "session"])

    final_files = {"all_sessions": final_all_file}

    for sess in ["premarket", "regular", "postmarket"]:
        sess_year_files = [x[sess] for x in yearly_outputs if sess in x]
        if sess_year_files:
            final_sess_file = asset_dir / f"{symbol}_{start_year}_{end_year}_1min_{sess}.csv"
            combine_files(sess_year_files, final_sess_file, ["minute_dt", "session"])
            final_files[sess] = final_sess_file

    for files in yearly_outputs:
        for f in files.values():
            f.unlink(missing_ok=True)

    del yearly_outputs
    gc.collect()

    print(f"Finished asset {symbol}")
    return final_files


def finalize_asset_outputs(symbol: str, start_year: int, end_year: int, base_output_dir: Path):
    symbol = str(symbol).upper().strip()
    asset_dir = base_output_dir / symbol
    asset_dir.mkdir(parents=True, exist_ok=True)

    final_regular_file = asset_dir / f"{symbol}_{start_year}_{end_year}_1min_regular.csv"
    final_all_file = asset_dir / f"{symbol}_{start_year}_{end_year}_1min_all_sessions.csv"
    if required_files_exist([final_all_file, final_regular_file]):
        print(f"Skipping completed asset {symbol}", flush=True)
        out = {
            "all_sessions": final_all_file,
            "regular": final_regular_file,
        }
        pre = asset_dir / f"{symbol}_{start_year}_{end_year}_1min_premarket.csv"
        post = asset_dir / f"{symbol}_{start_year}_{end_year}_1min_postmarket.csv"
        if file_has_data(pre):
            out["premarket"] = pre
        if file_has_data(post):
            out["postmarket"] = post
        return out

    yearly_outputs = []
    for year in range(start_year, end_year + 1):
        ensure_year_available_from_existing(symbol, year, base_output_dir)
        year_all_file = asset_dir / f"{symbol}_{year}_1min_all_sessions.csv"
        year_regular_file = asset_dir / f"{symbol}_{year}_1min_regular.csv"
        if not required_files_exist([year_all_file, year_regular_file]):
            continue

        files = {
            "all_sessions": year_all_file,
            "regular": year_regular_file,
        }
        year_premarket_file = asset_dir / f"{symbol}_{year}_1min_premarket.csv"
        year_postmarket_file = asset_dir / f"{symbol}_{year}_1min_postmarket.csv"
        if file_has_data(year_premarket_file):
            files["premarket"] = year_premarket_file
        if file_has_data(year_postmarket_file):
            files["postmarket"] = year_postmarket_file
        yearly_outputs.append(files)

    if not yearly_outputs:
        print(f"No yearly outputs available to finalize asset {symbol}", flush=True)
        return {}

    asset_all_files = [x["all_sessions"] for x in yearly_outputs if "all_sessions" in x]
    combine_files(asset_all_files, final_all_file, ["minute_dt", "session"])

    final_files = {"all_sessions": final_all_file}

    for sess in ["premarket", "regular", "postmarket"]:
        sess_year_files = [x[sess] for x in yearly_outputs if sess in x]
        if sess_year_files:
            final_sess_file = asset_dir / f"{symbol}_{start_year}_{end_year}_1min_{sess}.csv"
            combine_files(sess_year_files, final_sess_file, ["minute_dt", "session"])
            final_files[sess] = final_sess_file

    print(f"Finished asset {symbol}", flush=True)
    return final_files


def _process_one_year_worker(symbol: str, year: int, base_output_dir: str, ctm_table_index, wrds_username: str, wrds_password: str):
    symbol = str(symbol).upper().strip()
    year = int(year)
    base_output_dir = Path(base_output_dir)
    reused_files = ensure_year_available_from_existing(symbol, year, base_output_dir)
    if reused_files:
        return reused_files

    conn = create_wrds_connection(wrds_username, wrds_password, load_library_list=False)
    print_state = {
        "raw_printed": False,
        "processed_printed": False,
    }
    try:
        asset_dir = base_output_dir / symbol
        asset_dir.mkdir(parents=True, exist_ok=True)
        return process_one_year(
            conn=conn,
            ctm_table_index=ctm_table_index,
            symbol=symbol,
            year=year,
            asset_dir=asset_dir,
            print_state=print_state,
        )
    finally:
        conn.close()


def _process_one_asset_worker(symbol: str, start_year: int, end_year: int, base_output_dir: str, ctm_table_index, wrds_username: str, wrds_password: str):
    conn = create_wrds_connection(wrds_username, wrds_password, load_library_list=False)
    print_state = {
        "raw_printed": False,
        "processed_printed": False,
    }
    try:
        return process_one_asset(
            conn=conn,
            ctm_table_index=ctm_table_index,
            symbol=str(symbol).upper().strip(),
            start_year=start_year,
            end_year=end_year,
            base_output_dir=Path(base_output_dir),
            print_state=print_state,
        )
    finally:
        conn.close()


def process_assets(assets, start_year: int, end_year: int, max_workers: int | None = None):
    base_output_dir = Path(f"data_{start_year}_{end_year}")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    seen_assets = set()
    normalized_assets = []
    for symbol in assets:
        normalized = str(symbol).upper().strip()
        if not normalized or normalized in seen_assets:
            continue
        normalized_assets.append(normalized)
        seen_assets.add(normalized)
    assets = normalized_assets

    active_assets = []
    pending_tasks = []
    for symbol in assets:
        asset_dir = base_output_dir / symbol
        if final_output_files(asset_dir, symbol, start_year, end_year):
            print(f"Skipping completed asset {symbol}", flush=True)
            continue

        active_assets.append(symbol)
        for year in range(start_year, end_year + 1):
            ensure_year_available_from_existing(symbol, year, base_output_dir)
            if year_output_files(asset_dir, symbol, year):
                print(f"Skipping completed year {year} for {symbol}", flush=True)
                continue
            pending_tasks.append((symbol, year))

    if not pending_tasks:
        for symbol in active_assets:
            finalize_asset_outputs(symbol, start_year, end_year, base_output_dir)
        print("All requested asset-years already have required output files.", flush=True)
        return

    wrds_username, wrds_password = get_wrds_credentials()
    conn = create_wrds_connection(wrds_username, wrds_password)

    try:
        ctm_table_index = build_ctm_daily_table_index(conn)
    finally:
        conn.close()

    if max_workers is None:
        max_workers = int(os.environ.get("DATA_RETRIEVE_MAX_WORKERS", DEFAULT_MAX_WORKERS))
    max_workers = max(1, int(max_workers))
    if max_workers > DEFAULT_WRDS_MAX_WORKERS:
        print(
            f"Reducing max_workers from {max_workers} to WRDS_MAX_WORKERS={DEFAULT_WRDS_MAX_WORKERS} "
            "to avoid WRDS connection limits.",
            flush=True,
        )
        max_workers = DEFAULT_WRDS_MAX_WORKERS

    if max_workers == 1:
        conn = create_wrds_connection(wrds_username, wrds_password, load_library_list=False)
        print_state = {
            "raw_printed": False,
            "processed_printed": False,
        }
        try:
            for symbol, year in pending_tasks:
                asset_dir = base_output_dir / symbol
                process_one_year(conn, ctm_table_index, symbol, year, asset_dir, print_state)
                gc.collect()
        finally:
            conn.close()
        for symbol in active_assets:
            finalize_asset_outputs(symbol, start_year, end_year, base_output_dir)
        return

    print(
        f"Running multi-process year-level retrieval with max_workers={max_workers} "
        f"across {len(pending_tasks)} missing asset-year tasks",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _process_one_year_worker,
                symbol,
                year,
                str(base_output_dir),
                ctm_table_index,
                wrds_username,
                wrds_password,
            ): (symbol, year)
            for symbol, year in pending_tasks
        }

        for future in as_completed(future_map):
            symbol, year = future_map[future]
            try:
                future.result()
                print(f"[parallel] Completed {symbol} {year}", flush=True)
            except Exception as exc:
                print(f"[parallel] Task {symbol} {year} failed: {exc}", flush=True)
                raise

    for symbol in active_assets:
        finalize_asset_outputs(symbol, start_year, end_year, base_output_dir)


if __name__ == "__main__":
    start_year = 2016
    end_year = 2025
    max_workers = int(os.environ.get("DATA_RETRIEVE_MAX_WORKERS", DEFAULT_MAX_WORKERS))

    assets = ["META","AAPL","TSLA","AVGO","AMZN","GOOG","AMZN","MSFT","WMT","BRKA"]
    process_assets(
        assets=assets,
        start_year=start_year,
        end_year=end_year,
        max_workers=max_workers)
