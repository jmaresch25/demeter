import copy
import decimal
import json
import logging
import os
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Final, List, Tuple

import pandas as pd
from pydantic import BaseModel, TypeAdapter, ValidationError, field_validator

from demeter import MarketTypeEnum
from demeter.broker import BASE_FREQ
from demeter.data import CacheManager
from demeter._typing import DemeterError
from demeter.utils import console_text


_NUMERIC_COLUMNS_FOR_CACHE: Final[tuple[str, ...]] = (
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
)


def get_new_order_list(old: List[List[float]], used: List[List[float]]) -> List[List[float]]:
    to_update: List[List[float]] = copy.deepcopy(old)
    for x in used:
        for y in range(len(to_update)):
            if float(x[0]) == to_update[y][0]:
                to_update[y][1] -= float(x[1])
                break
    return to_update


def round_decimal(num: Any, exponent: int) -> Decimal:
    """
    Adjusting the number to a specific number of digits.

    :param num: number in any type, such as int/float/Decimal/str
    :param exponent: specific number of digits
    """
    if not isinstance(num, Decimal):
        num = Decimal(num)
    val: Decimal = num.quantize(Decimal(f"1e{exponent}"), rounding=decimal.ROUND_HALF_UP)
    if exponent > 0:
        val = val.quantize(Decimal(0))
    return val


def position_to_df(positions: Any) -> pd.DataFrame:
    pos_dict: dict[str, list[Any]] = {
        "instrument_name": [],
        "expiry_time": [],
        "strike_price": [],
        "type": [],
        "amount": [],
        "avg_buy_price": [],
        "buy_amount": [],
        "avg_sell_price": [],
        "sell_amount": [],
    }
    for _, v in positions.items():
        pos_dict["instrument_name"].append(console_text.format_value(v.instrument_name))
        pos_dict["expiry_time"].append(console_text.format_value(v.expiry_time))
        pos_dict["strike_price"].append(console_text.format_value(v.strike_price))
        pos_dict["type"].append(console_text.format_value(v.type))
        pos_dict["amount"].append(console_text.format_value(v.amount))
        pos_dict["avg_buy_price"].append(console_text.format_value(v.avg_buy_price))
        pos_dict["buy_amount"].append(console_text.format_value(v.buy_amount))
        pos_dict["avg_sell_price"].append(console_text.format_value(v.avg_sell_price))
        pos_dict["sell_amount"].append(console_text.format_value(v.sell_amount))
    return pd.DataFrame(pos_dict)


def decode_instrument(instrument_name: str) -> Tuple[str, datetime, int, str]:
    try:
        payload = _InstrumentNamePayload.from_instrument_name(instrument_name)
    except (ValueError, ValidationError) as exc:
        raise DemeterError(f"Invalid deribit instrument name: {instrument_name}") from exc
    return payload.token, payload.expiry_time, payload.strike_price, payload.option_type


def order_converter(array_str: str) -> List[List[float]]:
    try:
        parsed: Any = json.loads(array_str)
    except json.JSONDecodeError:
        return []
    try:
        validated_levels = _ORDER_LEVELS_ADAPTER.validate_python(parsed)
    except ValidationError:
        return []
    return [[float(price), float(amount)] for price, amount in validated_levels]


class _InstrumentNamePayload(BaseModel):
    token: str
    expiry_time: datetime
    strike_price: int
    option_type: str

    @field_validator("option_type")
    @classmethod
    def _validate_option_type(cls, value: str) -> str:
        if value not in {"PUT", "CALL"}:
            raise ValueError("option_type must be PUT or CALL")
        return value

    @classmethod
    def from_instrument_name(cls, instrument_name: str) -> "_InstrumentNamePayload":
        split = instrument_name.split("-")
        if len(split) != 4:
            raise ValueError("instrument name must have four parts")
        token, date_part, strike_part, kind_part = split
        option_type = "PUT" if kind_part == "P" else "CALL" if kind_part == "C" else kind_part
        expiry_time = datetime.strptime(f"{date_part} 08:00:00", "%d%b%y %H:%M:%S")
        return cls(
            token=token,
            expiry_time=expiry_time,
            strike_price=int(strike_part),
            option_type=option_type,
        )


_ORDER_LEVELS_ADAPTER = TypeAdapter(List[Tuple[float, float]])


def load_deribit_option_data(start_date: date, end_date: date, data_path: str) -> pd.DataFrame:
    """
    Load data from folder set in data_path. Those data file should be downloaded by demeter, and meet name rule.
    Deribit-option-book-{token}-{day.strftime('%Y%m%d')}.csv
    """
    logger: logging.Logger = logging.getLogger("Deribit data")

    cache_key: str = CacheManager.get_cache_key(
        MarketTypeEnum.deribit_option.name,
        start_date,
        end_date,
        address="ETH",
    )
    cache_df: Any = CacheManager.load(cache_key)
    if cache_df is not None and not cache_df.empty:
        return cache_df

    logger.info(f"{MarketTypeEnum.deribit_option.name} start load files from {start_date} to {end_date}...")

    from tqdm import tqdm

    day: date = start_date
    df: pd.DataFrame = pd.DataFrame()

    with tqdm(total=(end_date - start_date).days + 1, ncols=150) as pbar:
        while day <= end_date:
            path: str = os.path.join(
                data_path,
                f"Deribit-option-book-ETH-{day.strftime('%Y%m%d')}.csv",
            )
            if not os.path.exists(path):
                logging.warning(f"resource file {path} not found")
                day += timedelta(days=1)
                pbar.update()
                continue

            day_df: pd.DataFrame = pd.read_csv(
                str(path),
                parse_dates=["time", "expiry_time"],
                index_col=["time", "instrument_name"],
                converters={"asks": order_converter, "bids": order_converter},
                low_memory=False,
            )
            day_df["t"] = pd.to_timedelta(day_df["t"])
            day_df.drop(columns=["actual_time", "min_price", "max_price"], inplace=True, errors="ignore")

            df = pd.concat([df, day_df], axis=0)
            day += timedelta(days=1)
            pbar.update()

    df = df.sort_index()

    coerce_columns: list[str] = [c for c in _NUMERIC_COLUMNS_FOR_CACHE if c in df.columns]
    total_coerced_cells: int = 0
    total_non_empty_cells: int = 0

    for col in coerce_columns:
        raw: pd.Series = df[col]
        raw_str: pd.Series = raw.astype(str)
        non_empty_mask: pd.Series = raw.notna() & (raw_str.str.strip() != "")

        parsed: pd.Series = pd.to_numeric(raw, errors="coerce")
        bad_mask: pd.Series = non_empty_mask & parsed.isna()

        coerced_now: int = int(bad_mask.sum())
        non_empty_now: int = int(non_empty_mask.sum())
        total_coerced_cells += coerced_now
        total_non_empty_cells += non_empty_now

        if coerced_now > 0:
            bad_examples: list[str] = raw_str[bad_mask].head(5).tolist()
            logger.warning(
                "deribit_cache_numeric_coerce col=%s coerced=%d non_empty=%d examples=%s",
                str(col),
                int(coerced_now),
                int(non_empty_now),
                str(bad_examples),
            )

        df[col] = parsed

    ratio: float = float(total_coerced_cells) / float(total_non_empty_cells) if total_non_empty_cells > 0 else 0.0
    logger.info(
        "deribit_cache_numeric_coerce_summary coerced_cells=%d total_non_empty_cells=%d ratio=%.6f",
        int(total_coerced_cells),
        int(total_non_empty_cells),
        float(ratio),
    )

    CacheManager.save(cache_key, df)
    logger.info("data has been prepared")
    return df


def get_price_from_data(data: pd.DataFrame) -> pd.Series:
    """
    Get hourly underlying price.
    """
    price: list[dict[str, Any]] = []
    for hour, hour_df in data.groupby(level=0):
        price.append({"time": hour, "ETH": hour_df.iloc[0]["underlying_price"]})
    price_df: pd.DataFrame = pd.DataFrame(price)
    price_df.set_index(["time"], inplace=True)

    price_df.loc[price_df.tail(1).index[0].ceil("1d")] = 0
    price_df = price_df.resample(BASE_FREQ).ffill()
    return price_df.drop(price_df.index[-1])
