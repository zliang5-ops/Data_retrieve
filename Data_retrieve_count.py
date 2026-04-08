import gc
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from numba import njit
except Exception:  
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


from Data_retrieve import (
    append_csv,
    combine_files,
    create_wrds_connection,
    file_has_data,
    get_wrds_credentials,
    split_symbol,
    year_bimonths,
)


DEFAULT_MAX_WORKERS = max(1, int(os.environ.get("DATA_RETRIEVE_COUNT_MAX_WORKERS", "8")))
PRICE_TOLERANCE = float(os.environ.get("DATA_RETRIEVE_COUNT_PRICE_TOLERANCE", "1e-9"))
REGULAR_START = "09:30:00"
REGULAR_END = "16:00:00"
QUOTE_TABLE_PREFIXES = ("ctq_", "cqm_", "cq_")
TRADE_TABLE_PREFIX = "ctm_"


def build_daily_table_index(conn):
    tables = conn.list_tables(library="taqmsec")
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
        else:
            quote_prefix = next((prefix for prefix in QUOTE_TABLE_PREFIXES if name.startswith(prefix)), None)
            if quote_prefix is None or len(name) != len(quote_prefix) + 8:
                continue
            try:
                table_date = pd.to_datetime(name[-8:], format="%Y%m%d").date()
            except Exception:
                continue
            quote_tables[table_date] = table_name

    out = []
    for table_date in sorted(set(trade_tables) & set(quote_tables)):
        out.append((table_date, trade_tables[table_date], quote_tables[table_date]))
    return out


def get_daily_tables_in_range(table_index, start_date: date, end_date: date):
    return [(d, trade_t, quote_t) for d, trade_t, quote_t in table_index if start_date <= d <= end_date]


def fetch_raw_trades_one_day(conn, table_name: str, symbol: str) -> pd.DataFrame:
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
    return conn.raw_sql(sql, params={"sym_root": sym_root, "sym_suffix": sym_suffix})


def fetch_raw_quotes_one_day(conn, table_name: str, symbol: str) -> pd.DataFrame:
    sym_root, sym_suffix = split_symbol(symbol)
    sql = f"""
    select
        date,
        time_m,
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
    """
    return conn.raw_sql(sql, params={"sym_root": sym_root, "sym_suffix": sym_suffix})


def _normalize_time_column(df: pd.DataFrame, date_col: str, time_col: str, out_col: str):
    if df.empty:
        return df.iloc[0:0].copy()

    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce").dt.normalize()
    td = pd.to_timedelta(x[time_col].astype(str).str.strip(), errors="coerce")
    mask = x[date_col].notna() & td.notna()
    if not mask.all():
        x = x.loc[mask].copy()
        td = td.loc[mask]

    if x.empty:
        return x

    x[out_col] = x[date_col] + td
    x["minute_dt"] = x[out_col].dt.floor("min")
    hhmm = x[out_col].dt.hour * 100 + x[out_col].dt.minute
    x = x.loc[(hhmm >= 930) & (hhmm < 1600)].copy()
    return x


def prepare_trade_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df.iloc[0:0].copy()

    x = raw_df.loc[:, ["date", "time_m", "price", "size"]].copy()
    x["price"] = pd.to_numeric(x["price"], errors="coerce")
    x["size"] = pd.to_numeric(x["size"], errors="coerce")
    x = x.loc[(x["price"] > 0) & (x["size"] > 0)].copy()
    x = _normalize_time_column(x, "date", "time_m", "trade_dt")
    if x.empty:
        return x
    x = x.sort_values("trade_dt", kind="mergesort").reset_index(drop=True)
    return x


def prepare_quote_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df.iloc[0:0].copy()

    x = raw_df.loc[:, ["date", "time_m", "bid", "bidsiz", "ask", "asksiz"]].copy()
    x["bid"] = pd.to_numeric(x["bid"], errors="coerce")
    x["ask"] = pd.to_numeric(x["ask"], errors="coerce")
    x["bidsiz"] = pd.to_numeric(x["bidsiz"], errors="coerce")
    x["asksiz"] = pd.to_numeric(x["asksiz"], errors="coerce")
    x = x.loc[(x["bid"] > 0) & (x["ask"] > 0) & (x["ask"] >= x["bid"])].copy()
    x = _normalize_time_column(x, "date", "time_m", "quote_dt")
    if x.empty:
        return x
    x = x.sort_values("quote_dt", kind="mergesort").reset_index(drop=True)
    return x


@njit(cache=True)
def classify_trade_volumes_numba(trade_times_ns, trade_prices, trade_sizes, quote_times_ns, bids, asks, price_tolerance):
    n = trade_times_ns.shape[0]
    total = np.zeros(n, dtype=np.float64)
    above = np.zeros(n, dtype=np.float64)
    below = np.zeros(n, dtype=np.float64)
    at_mid = np.zeros(n, dtype=np.float64)
    matched = np.zeros(n, dtype=np.uint8)

    quote_idx = 0
    have_quote = False
    current_bid = 0.0
    current_ask = 0.0

    for i in range(n):
        trade_time = trade_times_ns[i]
        while quote_idx < quote_times_ns.shape[0] and quote_times_ns[quote_idx] <= trade_time:
            current_bid = bids[quote_idx]
            current_ask = asks[quote_idx]
            have_quote = True
            quote_idx += 1

        if not have_quote:
            continue

        midpoint = 0.5 * (current_bid + current_ask)
        size = trade_sizes[i]
        price = trade_prices[i]

        total[i] = size
        matched[i] = 1

        if price > midpoint + price_tolerance:
            above[i] = size
        elif price < midpoint - price_tolerance:
            below[i] = size
        else:
            at_mid[i] = size

    return matched, total, above, below, at_mid


def build_regular_minute_grid(trading_day: pd.Timestamp) -> pd.DataFrame:
    start = pd.Timestamp(f"{trading_day.date()} {REGULAR_START}")
    end = pd.Timestamp(f"{trading_day.date()} 15:59:00")
    return pd.DataFrame({"minute_dt": pd.date_range(start=start, end=end, freq="min")})


def build_minute_counts_one_day(trades_df: pd.DataFrame, quotes_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    trades = prepare_trade_data(trades_df)
    quotes = prepare_quote_data(quotes_df)

    if quotes.empty:
        return pd.DataFrame()

    trading_day = quotes["date"].iloc[0] if "date" in quotes.columns else quotes["quote_dt"].iloc[0].normalize()
    minute_grid = build_regular_minute_grid(trading_day)

    minute_quotes = (
        quotes.sort_values("quote_dt", kind="mergesort")
        .groupby("minute_dt", sort=True, as_index=False)[["bid", "bidsiz", "ask", "asksiz"]]
        .last()
    )

    out = minute_grid.merge(minute_quotes, on="minute_dt", how="left", sort=True)
    out[["bid", "bidsiz", "ask", "asksiz"]] = out[["bid", "bidsiz", "ask", "asksiz"]].ffill()
    out["bidsiz"] = pd.array(np.round(out["bidsiz"]), dtype="Int64")
    out["asksiz"] = pd.array(np.round(out["asksiz"]), dtype="Int64")

    if not trades.empty:
        trade_times_ns = trades["trade_dt"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        trade_prices = trades["price"].to_numpy(dtype=np.float64)
        trade_sizes = trades["size"].to_numpy(dtype=np.float64)
        quote_times_ns = quotes["quote_dt"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        bids = quotes["bid"].to_numpy(dtype=np.float64)
        asks = quotes["ask"].to_numpy(dtype=np.float64)

        matched, total, above, below, at_mid = classify_trade_volumes_numba(
            trade_times_ns,
            trade_prices,
            trade_sizes,
            quote_times_ns,
            bids,
            asks,
            PRICE_TOLERANCE,
        )

        matched_mask = matched.astype(bool)
        if matched_mask.any():
            classified = pd.DataFrame(
                {
                    "minute_dt": trades.loc[matched_mask, "minute_dt"].to_numpy(),
                    "trade_volume_total": total[matched_mask],
                    "trade_volume_above_mid": above[matched_mask],
                    "trade_volume_below_mid": below[matched_mask],
                    "trade_volume_at_mid": at_mid[matched_mask],
                }
            )
            minute_trade_counts = classified.groupby("minute_dt", sort=True, as_index=False).sum()
            out = out.merge(minute_trade_counts, on="minute_dt", how="left", sort=True)

    for col in [
        "trade_volume_total",
        "trade_volume_above_mid",
        "trade_volume_below_mid",
        "trade_volume_at_mid",
    ]:
        if col not in out.columns:
            out[col] = 0.0
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    sym_root, sym_suffix = split_symbol(symbol)
    out["date"] = out["minute_dt"].dt.date
    out["time_m"] = out["minute_dt"].dt.time
    out["sym_root"] = sym_root
    out["sym_suffix"] = sym_suffix
    out["session"] = "regular"

    out = out[
        [
            "minute_dt",
            "date",
            "time_m",
            "sym_root",
            "sym_suffix",
            "session",
            "bid",
            "bidsiz",
            "ask",
            "asksiz",
            "trade_volume_total",
            "trade_volume_above_mid",
            "trade_volume_below_mid",
            "trade_volume_at_mid",
        ]
    ].sort_values("minute_dt", kind="mergesort")

    return out.reset_index(drop=True)


def count_year_output_file(asset_dir: Path, symbol: str, year: int) -> Path:
    return asset_dir / f"{symbol}_{year}_1min_count_regular.csv"


def count_final_output_file(asset_dir: Path, symbol: str, start_year: int, end_year: int) -> Path:
    return asset_dir / f"{symbol}_{start_year}_{end_year}_1min_count_regular.csv"


def process_one_day(conn, trade_table: str, quote_table: str, symbol: str):
    trades_df = fetch_raw_trades_one_day(conn, trade_table, symbol)
    quotes_df = fetch_raw_quotes_one_day(conn, quote_table, symbol)
    try:
        return build_minute_counts_one_day(trades_df, quotes_df, symbol)
    finally:
        del trades_df
        del quotes_df
        gc.collect()


def process_one_bimonth(conn, table_index, symbol: str, year: int, start_date: date, end_date: date, block_label: str, asset_dir: Path):
    print(f"Processing {symbol} {year} {block_label}: {start_date} to {end_date}", flush=True)
    daily_tables = get_daily_tables_in_range(table_index, start_date, end_date)
    if not daily_tables:
        print(f"No trade/quote tables found for {symbol} {year} {block_label}", flush=True)
        return None

    block_file = asset_dir / f"{symbol}_{year}_{block_label}_1min_count_regular.csv"
    if file_has_data(block_file):
        print(f"Skipping completed block {symbol} {year} {block_label}", flush=True)
        return block_file

    daily_tmp_files = []
    for trading_day, trade_table, quote_table in daily_tables:
        day_tmp = asset_dir / f"{symbol}_{year}_{block_label}_{trading_day.strftime('%Y%m%d')}_1min_count_tmp.csv"
        daily_tmp_files.append(day_tmp)
        if file_has_data(day_tmp):
            continue

        day_df = process_one_day(conn, trade_table, quote_table, symbol)
        if day_df.empty:
            continue
        append_csv(day_df, day_tmp)
        del day_df
        gc.collect()

    available_daily_files = [path for path in daily_tmp_files if file_has_data(path)]
    if not available_daily_files:
        print(f"No daily output created for {symbol} {year} {block_label}", flush=True)
        return None

    combine_files(available_daily_files, block_file, ["minute_dt"])
    for tmp_file in available_daily_files:
        tmp_file.unlink(missing_ok=True)
    return block_file


def process_one_year(conn, table_index, symbol: str, year: int, asset_dir: Path):
    year_file = count_year_output_file(asset_dir, symbol, year)
    if file_has_data(year_file):
        print(f"Skipping completed year {year} for {symbol}", flush=True)
        return year_file

    block_files = []
    for start_date, end_date, block_label in year_bimonths(year):
        block_file = process_one_bimonth(conn, table_index, symbol, year, start_date, end_date, block_label, asset_dir)
        if block_file is not None and file_has_data(block_file):
            block_files.append(block_file)
        gc.collect()

    if not block_files:
        print(f"No yearly count output for {symbol} {year}", flush=True)
        return None

    combine_files(block_files, year_file, ["minute_dt"])
    for block_file in block_files:
        block_file.unlink(missing_ok=True)

    print(f"Finished year {year} for {symbol}", flush=True)
    return year_file


def finalize_asset_outputs(symbol: str, start_year: int, end_year: int, base_output_dir: Path):
    symbol = str(symbol).upper().strip()
    asset_dir = base_output_dir / symbol
    asset_dir.mkdir(parents=True, exist_ok=True)

    final_file = count_final_output_file(asset_dir, symbol, start_year, end_year)
    if file_has_data(final_file):
        print(f"Skipping completed asset {symbol}", flush=True)
        return final_file

    year_files = []
    for year in range(start_year, end_year + 1):
        year_file = count_year_output_file(asset_dir, symbol, year)
        if file_has_data(year_file):
            year_files.append(year_file)

    if not year_files:
        print(f"No yearly outputs available to finalize asset {symbol}", flush=True)
        return None

    combine_files(year_files, final_file, ["minute_dt"])
    print(f"Finished asset {symbol}", flush=True)
    return final_file


def _process_one_year_worker(symbol: str, year: int, base_output_dir: str, table_index, wrds_username: str, wrds_password: str):
    symbol = str(symbol).upper().strip()
    asset_dir = Path(base_output_dir) / symbol
    asset_dir.mkdir(parents=True, exist_ok=True)

    conn = create_wrds_connection(wrds_username, wrds_password, load_library_list=False)
    try:
        return process_one_year(conn, table_index, symbol, year, asset_dir)
    finally:
        conn.close()


def process_assets(assets, start_year: int, end_year: int, max_workers: int | None = None):
    base_output_dir = Path(f"data_{start_year}_{end_year}_count")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    seen_assets = set()
    normalized_assets = []
    for symbol in assets:
        normalized = str(symbol).upper().strip()
        if not normalized or normalized in seen_assets:
            continue
        normalized_assets.append(normalized)
        seen_assets.add(normalized)

    pending_tasks = []
    active_assets = []
    for symbol in normalized_assets:
        asset_dir = base_output_dir / symbol
        asset_dir.mkdir(parents=True, exist_ok=True)
        final_file = count_final_output_file(asset_dir, symbol, start_year, end_year)
        if file_has_data(final_file):
            print(f"Skipping completed asset {symbol}", flush=True)
            continue

        active_assets.append(symbol)
        for year in range(start_year, end_year + 1):
            year_file = count_year_output_file(asset_dir, symbol, year)
            if file_has_data(year_file):
                print(f"Skipping completed year {year} for {symbol}", flush=True)
                continue
            pending_tasks.append((symbol, year))

    if not pending_tasks:
        for symbol in active_assets:
            finalize_asset_outputs(symbol, start_year, end_year, base_output_dir)
        print("All requested asset-years already have required count files.", flush=True)
        return

    wrds_username, wrds_password = get_wrds_credentials()
    conn = create_wrds_connection(wrds_username, wrds_password)
    try:
        table_index = build_daily_table_index(conn)
    finally:
        conn.close()

    if max_workers is None:
        max_workers = DEFAULT_MAX_WORKERS
    max_workers = max(1, int(max_workers))

    if max_workers == 1:
        conn = create_wrds_connection(wrds_username, wrds_password, load_library_list=False)
        try:
            for symbol, year in pending_tasks:
                asset_dir = base_output_dir / symbol
                process_one_year(conn, table_index, symbol, year, asset_dir)
                gc.collect()
        finally:
            conn.close()
    else:
        print(
            f"Running multi-process count retrieval with max_workers={max_workers} "
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
                    table_index,
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
    max_workers = int(os.environ.get("DATA_RETRIEVE_COUNT_MAX_WORKERS", str(DEFAULT_MAX_WORKERS)))

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
    ]

    process_assets(
        assets=assets,
        start_year=start_year,
        end_year=end_year,
        max_workers=max_workers,
    )
