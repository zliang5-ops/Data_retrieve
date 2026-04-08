# WRDS TAQ Retrieval

This project contains two retrieval scripts built on WRDS TAQ millisecond data:

- `Data_retrieve.py`: downloads trade data and builds 1-minute OHLCV bars.
- `Data_retrieve_count.py`: matches each trade to the most recent quote in event time, classifies trade volume as above / at / below the midpoint, then aggregates to 1-minute output.

Both scripts are resumable, work year-by-year in bimonth blocks, and write CSVs by ticker.



## Credentials

Both scripts use the same WRDS login flow

```powershell
$env:WRDS_USERNAME="jerryliang"
$env:WRDS_PASSWORD="Jerry200385241/"
```

## Data_retrieve.py

Purpose:
- Pull TAQ trades for each symbol and convert them into 1-minute bars.
- Split output into `premarket`, `regular`, `postmarket`, and `all_sessions`.

Main output location:
- `data_<start_year>_<end_year>\<TICKER>\`

Typical files:
- `<TICKER>_<year>_1min_regular.csv`
- `<TICKER>_<year>_1min_all_sessions.csv`
- `<TICKER>_<start_year>_<end_year>_1min_regular.csv`
- `<TICKER>_<start_year>_<end_year>_1min_all_sessions.csv`

Key columns:
- `minute_dt`, `date`, `minute`, `session`
- `open`, `high`, `low`, `close`
- `volume`, `vwap`, `px_at_highest_volume`

## Data_retrieve_count.py

Purpose:
- Pull TAQ trades and quotes for each symbol.
- Match each trade to the latest quote with `quote_time <= trade_time`.
- Use the live midpoint at that event time to classify trade size into:
  - `trade_volume_above_mid`
  - `trade_volume_at_mid`
  - `trade_volume_below_mid`
- Aggregate those classified trade volumes into 1-minute regular-session rows ((regular session only)).

Main output location:
- `data_<start_year>_<end_year>_count\<TICKER>\`

Typical files:
- `<TICKER>_<year>_1min_count_regular.csv`
- `<TICKER>_<start_year>_<end_year>_1min_count_regular.csv`

Key columns:
- `minute_dt`, `date`, `time_m`, `session`
- `bid`, `bidsiz`, `ask`, `asksiz`
- `trade_volume_total`
- `trade_volume_above_mid`, `trade_volume_at_mid`, `trade_volume_below_mid`

Important note:
- Matching is done on the raw event timestamps from WRDS before minute aggregation.
- The final dataset is still 1-minute level because the classified trade volumes are summed by minute after matching.

## How The Pipeline Runs

1. Normalize the ticker symbol and create the ticker output folder.
2. Build the list of daily WRDS tables available in the requested year range.
3. Process each year in six bimonth blocks to keep jobs manageable.
4. Save block-level CSVs, combine them into yearly files, then combine yearly files into final multi-year files.
5. Skip files that already exist so interrupted runs can resume.

## Parallelism

- `Data_retrieve.py` uses `DATA_RETRIEVE_MAX_WORKERS`
- `Data_retrieve_count.py` uses `DATA_RETRIEVE_COUNT_MAX_WORKERS`

Example:

```powershell
$env:DATA_RETRIEVE_MAX_WORKERS="2"
$env:DATA_RETRIEVE_COUNT_MAX_WORKERS="8"
```

`Data_retrieve.py` is conservative by default because WRDS connection limits can be tight. `Data_retrieve_count.py` defaults to `8` workers unless overridden.


## Output Conventions

- Each ticker has its own subfolder.
- Year files are intermediate durable outputs.
- Multi-year files are the final consolidated outputs.
- Temporary daily or block files may appear during an active run and are removed after successful combination.

