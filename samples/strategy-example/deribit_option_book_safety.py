from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

import pandas as pd
from tenacity import RetryCallState, retry, stop_after_attempt, wait_exponential


LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)s %(message)s"

EXPECTED_COLUMNS: Final[list[str]] = [
    "instrument_name",
    "time",
    "actual_time",
    "state",
    "type",
    "strike_price",
    "t",
    "expiry_time",
    "vega",
    "theta",
    "rho",
    "gamma",
    "delta",
    "underlying_price",
    "settlement_price",
    "min_price",
    "max_price",
    "mark_price",
    "mark_iv",
    "last_price",
    "interest_rate",
    "bid_iv",
    "best_bid_price",
    "best_bid_amount",
    "ask_iv",
    "best_ask_price",
    "best_ask_amount",
    "asks",
    "bids",
]

NUMERIC_COLUMNS: Final[list[str]] = [
    "strike_price",
    "vega",
    "theta",
    "rho",
    "gamma",
    "delta",
    "underlying_price",
    "settlement_price",
    "min_price",
    "max_price",
    "mark_price",
    "mark_iv",
    "last_price",
    "interest_rate",
    "bid_iv",
    "best_bid_price",
    "best_bid_amount",
    "ask_iv",
    "best_ask_price",
    "best_ask_amount",
]


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format=LOG_FORMAT)


def _log_retry(retry_state: RetryCallState) -> None:
    logger: logging.Logger = logging.getLogger(__name__)
    attempt: int = int(retry_state.attempt_number)
    sleep: float = float(retry_state.next_action.sleep if retry_state.next_action else 0.0)
    logger.warning("retry attempt=%d sleep=%.2fs", attempt, sleep)


@dataclass(frozen=True, slots=True)
class SafetyReport:
    checked_files: int
    checked_rows_sampled_total: int
    repaired_files: int


def _date_range_inclusive(start_date: date, end_date: date) -> list[date]:
    days: list[date] = []
    cursor: date = start_date
    while cursor <= end_date:
        days.append(cursor)
        cursor = cursor + timedelta(days=1)
    return days


def _expected_daily_filename(underlying: str, day: date) -> str:
    return f"Deribit-option-book-{underlying}-{day.strftime('%Y%m%d')}.csv"


@retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=5.0), stop=stop_after_attempt(5), before_sleep=_log_retry)
def _read_csv_sample(path: Path, nrows: int) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(path, low_memory=False, nrows=nrows)
    return df


def _assert_exact_columns(df: pd.DataFrame, path: Path) -> None:
    got: list[str] = [str(c) for c in df.columns.tolist()]
    if got != EXPECTED_COLUMNS:
        raise ValueError(
            f"schema_mismatch path={str(path)!r} "
            f"expected_cols={EXPECTED_COLUMNS!r} got_cols={got!r}"
        )


def _assert_parseable_times(df: pd.DataFrame, path: Path) -> None:
    t_time: pd.Series = pd.to_datetime(df["time"], errors="coerce")
    t_actual: pd.Series = pd.to_datetime(df["actual_time"], errors="coerce")
    t_expiry: pd.Series = pd.to_datetime(df["expiry_time"], errors="coerce")

    if t_time.isna().any():
        raise ValueError(f"invalid_time_column path={str(path)!r} col='time'")
    if t_actual.isna().any():
        raise ValueError(f"invalid_time_column path={str(path)!r} col='actual_time'")
    if t_expiry.isna().any():
        raise ValueError(f"invalid_time_column path={str(path)!r} col='expiry_time'")

    td: pd.Series = t_expiry - t_time
    if td.isna().any():
        raise ValueError(f"invalid_timedelta path={str(path)!r} t=expiry_time-time produced NaT")


def _assert_enums(df: pd.DataFrame, path: Path) -> None:
    option_type_values: set[str] = set(df["type"].dropna().astype(str).unique().tolist())
    if not option_type_values.issubset({"CALL", "PUT"}):
        raise ValueError(f"invalid_type_values path={str(path)!r} values={sorted(option_type_values)!r}")

    state_values: set[str] = set(df["state"].dropna().astype(str).unique().tolist())
    if not state_values.issubset({"open"}):
        raise ValueError(f"invalid_state_values path={str(path)!r} values={sorted(state_values)!r}")


def _assert_numeric_convertible(df: pd.DataFrame, path: Path) -> None:
    for col in NUMERIC_COLUMNS:
        raw: pd.Series = df[col]
        parsed: pd.Series = pd.to_numeric(raw, errors="coerce")
        non_empty_mask: pd.Series = raw.notna() & (raw.astype(str).str.strip() != "")
        bad_mask: pd.Series = non_empty_mask & parsed.isna()
        if bool(bad_mask.any()):
            bad_examples: list[str] = raw[bad_mask].astype(str).head(5).tolist()
            raise ValueError(f"non_numeric_value path={str(path)!r} col={col!r} examples={bad_examples!r}")


def _parse_orderbook_side(value: Any) -> list[list[float]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        s: str = value.strip()
        if s == "" or s == "[]":
            return []
        parsed: Any = json.loads(s)
    else:
        parsed = value

    if parsed == []:
        return []
    if not isinstance(parsed, list):
        raise ValueError(f"orderbook_not_list parsed_type={type(parsed).__name__}")

    out: list[list[float]] = []
    for level in parsed:
        if not isinstance(level, (list, tuple)) or len(level) != 2:
            raise ValueError("orderbook_level_invalid_shape")
        p_raw: Any = level[0]
        a_raw: Any = level[1]
        p: float = float(p_raw)
        a: float = float(a_raw)
        out.append([p, a])
    return out


def _assert_orderbook_json(df: pd.DataFrame, path: Path) -> None:
    asks: pd.Series = df["asks"]
    bids: pd.Series = df["bids"]

    for side_name, side_series in (("asks", asks), ("bids", bids)):
        for idx, v in enumerate(side_series.head(50).tolist()):
            try:
                _ = _parse_orderbook_side(v)
            except Exception as exc:
                raise ValueError(f"invalid_orderbook_json path={str(path)!r} side={side_name!r} row={idx} err={str(exc)!r}") from exc

# deribit_option_book_safety.py

MAX_COERCED_RATIO_DEFAULT: Final[float] = 0.002  # ADD
COERCE_SENTINEL_PREFIXES: Final[tuple[str, ...]] = ("SYN.",)  # ADD


# deribit_option_book_safety.py

def _coerce_numeric_columns_inplace(  # REPLACE signature + body
    df: pd.DataFrame,
    path: Path,
    logger: logging.Logger,
    max_coerced_ratio: float,
) -> None:
    total_non_empty_cells: int = 0
    coerced_cells: int = 0

    for col in NUMERIC_COLUMNS:
        raw: pd.Series = df[col]
        raw_str: pd.Series = raw.astype(str)
        non_empty_mask: pd.Series = raw.notna() & (raw_str.str.strip() != "")
        total_non_empty_cells += int(non_empty_mask.sum())

        parsed: pd.Series = pd.to_numeric(raw, errors="coerce")
        bad_mask: pd.Series = non_empty_mask & parsed.isna()

        if bool(bad_mask.any()):
            bad_values: list[str] = raw_str[bad_mask].head(5).tolist()
            has_syn_like: bool = any(v.startswith(COERCE_SENTINEL_PREFIXES) for v in bad_values)

            coerced_now: int = int(bad_mask.sum())
            coerced_cells += coerced_now
            df[col] = parsed

            logger.warning(
                "safety_numeric_coerce path=%s col=%s coerced=%d examples=%s syn_like=%s",
                str(path),
                str(col),
                int(coerced_now),
                str(bad_values),
                str(has_syn_like),
            )
        else:
            df[col] = parsed

    if total_non_empty_cells <= 0:
        logger.info(
            "safety_numeric_coerce_summary path=%s coerced_cells=%d total_non_empty_cells=%d ratio=0.000000 max_ratio=%.6f",
            str(path),
            int(coerced_cells),
            int(total_non_empty_cells),
            float(max_coerced_ratio),
        )
        return

    ratio: float = float(coerced_cells) / float(total_non_empty_cells)
    logger.info(
        "safety_numeric_coerce_summary path=%s coerced_cells=%d total_non_empty_cells=%d ratio=%.6f max_ratio=%.6f",
        str(path),
        int(coerced_cells),
        int(total_non_empty_cells),
        float(ratio),
        float(max_coerced_ratio),
    )
    if ratio > float(max_coerced_ratio):
        raise ValueError(
            f"too_many_numeric_coercions path={str(path)!r} "
            f"ratio={ratio:.6f} max_ratio={float(max_coerced_ratio):.6f}"
        )





def validate_deribit_option_book_files(
    data_path: str | Path,
    start_date: date,
    end_date: date,
    underlying: str,
    sample_rows_per_file: int = 500,
    strict_numeric: bool = True,  # ADD
    max_coerced_ratio: float = MAX_COERCED_RATIO_DEFAULT,  # ADD
) -> SafetyReport:
    logger: logging.Logger = logging.getLogger(__name__)
    data_dir: Path = Path(data_path).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"data_path_not_found {str(data_dir)!r}")

    days: list[date] = _date_range_inclusive(start_date, end_date)
    checked_files: int = 0
    checked_rows_sampled_total: int = 0

    logger.info(
        "safety_start data_path=%s underlying=%s start=%s end=%s sample_rows_per_file=%d",
        str(data_dir),
        str(underlying),
        start_date.isoformat(),
        end_date.isoformat(),
        int(sample_rows_per_file),
    )

    for day in days:
        fname: str = _expected_daily_filename(underlying=underlying, day=day)
        fpath: Path = data_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"missing_daily_file {str(fpath)!r}")

        df_sample: pd.DataFrame = _read_csv_sample(fpath, nrows=int(sample_rows_per_file))
        if df_sample.empty:
            raise ValueError(f"empty_file {str(fpath)!r}")

        _assert_exact_columns(df_sample, fpath)
        _assert_parseable_times(df_sample, fpath)
        _assert_enums(df_sample, fpath)

        if strict_numeric:  # ADD
            _assert_numeric_convertible(df_sample, fpath)  # ADD
        else:  # ADD
            _coerce_numeric_columns_inplace(  # ADD
                df=df_sample,
                path=fpath,
                logger=logger,
                max_coerced_ratio=float(max_coerced_ratio),
            )
        _assert_orderbook_json(df_sample, fpath)

        checked_files += 1
        checked_rows_sampled_total += int(len(df_sample))

        logger.info(
            "safety_file_ok path=%s rows_sampled=%d",
            str(fpath),
            int(len(df_sample)),
        )

    logger.info(
        "safety_done checked_files=%d rows_sampled_total=%d",
        int(checked_files),
        int(checked_rows_sampled_total),
    )

    report: SafetyReport = SafetyReport(
        checked_files=int(checked_files),
        checked_rows_sampled_total=int(checked_rows_sampled_total),
        repaired_files=0,
    )
    return report
