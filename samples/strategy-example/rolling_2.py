from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Final, Optional

import pandas as pd

from demeter import (
    Actuator,
    DemeterError,
    MarketInfo,
    MarketTypeEnum,
    Snapshot,
    Strategy,
)
from demeter.deribit import DeribitOptionMarket, get_price_from_data, load_deribit_option_data

# ADD: safety validator (same workaround flow: validate -> coerce/skip strict -> proceed)
from deribit_option_book_safety import MAX_COERCED_RATIO_DEFAULT, validate_deribit_option_book_files  # type: ignore


logger: logging.Logger = logging.getLogger(__name__)

pd.options.display.max_columns = None
pd.set_option("display.width", 5000)

market_key: MarketInfo = MarketInfo("option_test", MarketTypeEnum.deribit_option)

# ADD: exact same numeric columns family that causes ArrowInvalid when cache tries to feather
_NUMERIC_COLUMNS_FOR_ARROW: Final[tuple[str, ...]] = (
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


def _coerce_numeric_columns_for_arrow(
    df: pd.DataFrame,
    *,
    max_coerced_ratio: float,
) -> pd.DataFrame:
    # ADD: hardening stage to prevent pyarrow/lib cache failures when any numeric column is "object" w/ strings
    if df is None or df.empty:
        return df

    cols: list[str] = [c for c in _NUMERIC_COLUMNS_FOR_ARROW if c in df.columns]
    if not cols:
        return df

    total_coerced_cells: int = 0
    total_non_empty_cells: int = 0

    for col in cols:
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
                "arrow_numeric_coerce col=%s coerced=%d non_empty=%d examples=%s",
                str(col),
                int(coerced_now),
                int(non_empty_now),
                str(bad_examples),
            )

        df[col] = parsed

    ratio: float = float(total_coerced_cells) / float(total_non_empty_cells) if total_non_empty_cells > 0 else 0.0
    logger.info(
        "arrow_numeric_coerce_summary coerced_cells=%d total_non_empty_cells=%d ratio=%.6f max_ratio=%.6f",
        int(total_coerced_cells),
        int(total_non_empty_cells),
        float(ratio),
        float(max_coerced_ratio),
    )

    if ratio > float(max_coerced_ratio):
        raise ValueError(
            f"too_many_numeric_coercions_for_arrow ratio={ratio:.6f} max_ratio={float(max_coerced_ratio):.6f}"
        )

    return df


@dataclass(frozen=True)
class IronCondorLeg:
    instrument_name: str
    strike_price: int
    option_type: str  # "PUT" | "CALL"
    side: str  # "short" | "long"


@dataclass
class IronCondorState:
    expiry_time: pd.Timestamp
    amount: Decimal
    short_put: IronCondorLeg
    long_put: IronCondorLeg
    short_call: IronCondorLeg
    long_call: IronCondorLeg


class RollingIronCondorStrategy(Strategy):
    def __init__(
        self,
        *,
        amount_multiplier: Decimal = Decimal("1"),
        short_distance_pct: Decimal = Decimal("0.05"),
        wing_distance_pct: Decimal = Decimal("0.10"),
        min_dte_days: int = 7,
        max_dte_days: int = 35,
    ) -> None:
        super().__init__()
        self.amount_multiplier: Decimal = amount_multiplier
        self.short_distance_pct: Decimal = short_distance_pct
        self.wing_distance_pct: Decimal = wing_distance_pct
        self.min_dte_days: int = min_dte_days
        self.max_dte_days: int = max_dte_days
        self.condor_state: Optional[IronCondorState] = None

    def initialize(self) -> None:
        return

    def on_bar(self, snapshot: Snapshot) -> None:
        market: DeribitOptionMarket = self.broker.markets.default

        ts: pd.Timestamp = pd.Timestamp(snapshot.timestamp)
        if ts != ts.floor("1h"):
            return

        orderbook_df: pd.DataFrame = market.market_status.data
        if orderbook_df is None or orderbook_df.empty:
            return

        spot_usd: Decimal = Decimal(str(orderbook_df.iloc[0]["underlying_price"]))

        if self.condor_state is None:
            self._open_new_condor(market=market, orderbook_df=orderbook_df, now=ts, spot_usd=spot_usd)
            return

        if not self._condor_positions_still_exist(market=market, state=self.condor_state):
            self.condor_state = None
            return

        short_put_k: Decimal = Decimal(self.condor_state.short_put.strike_price)
        short_call_k: Decimal = Decimal(self.condor_state.short_call.strike_price)
        if not (spot_usd <= short_put_k or spot_usd >= short_call_k):
            return

        self._close_condor(market=market, state=self.condor_state)
        self.condor_state = None
        self._open_new_condor(market=market, orderbook_df=orderbook_df, now=ts, spot_usd=spot_usd)

    def notify(self, action: object) -> None:
        logger.info("%s", action)

    def _open_new_condor(
        self,
        *,
        market: DeribitOptionMarket,
        orderbook_df: pd.DataFrame,
        now: pd.Timestamp,
        spot_usd: Decimal,
    ) -> None:
        expiry_time: Optional[pd.Timestamp] = self._select_expiry(orderbook_df=orderbook_df, now=now)
        if expiry_time is None:
            return

        expiry_df: pd.DataFrame = orderbook_df.loc[orderbook_df["expiry_time"] == expiry_time].copy()
        if expiry_df.empty:
            return

        puts_df: pd.DataFrame = expiry_df.loc[expiry_df["type"] == "PUT"]
        calls_df: pd.DataFrame = expiry_df.loc[expiry_df["type"] == "CALL"]
        if puts_df.empty or calls_df.empty:
            return

        sp_target: Decimal = spot_usd * (Decimal("1") - self.short_distance_pct)
        lp_target: Decimal = spot_usd * (Decimal("1") - self.wing_distance_pct)
        sc_target: Decimal = spot_usd * (Decimal("1") + self.short_distance_pct)
        lc_target: Decimal = spot_usd * (Decimal("1") + self.wing_distance_pct)

        short_put_strike: Optional[int] = self._choose_put_strike_at_or_below(puts_df=puts_df, target=sp_target, spot=spot_usd)
        long_put_strike: Optional[int] = self._choose_put_strike_below(puts_df=puts_df, target=lp_target, below=short_put_strike)
        short_call_strike: Optional[int] = self._choose_call_strike_at_or_above(calls_df=calls_df, target=sc_target, spot=spot_usd)
        long_call_strike: Optional[int] = self._choose_call_strike_above(calls_df=calls_df, target=lc_target, above=short_call_strike)

        if short_put_strike is None or long_put_strike is None or short_call_strike is None or long_call_strike is None:
            return

        short_put_name: Optional[str] = self._instrument_for_strike(df=puts_df, strike_price=short_put_strike)
        long_put_name: Optional[str] = self._instrument_for_strike(df=puts_df, strike_price=long_put_strike)
        short_call_name: Optional[str] = self._instrument_for_strike(df=calls_df, strike_price=short_call_strike)
        long_call_name: Optional[str] = self._instrument_for_strike(df=calls_df, strike_price=long_call_strike)

        if short_put_name is None or long_put_name is None or short_call_name is None or long_call_name is None:
            return

        base_amount: Decimal = market.token_config.min_amount
        amount: Decimal = base_amount * self.amount_multiplier

        market.sell(short_put_name, amount)
        market.sell(short_call_name, amount)
        market.buy(long_put_name, amount)
        market.buy(long_call_name, amount)

        self.condor_state = IronCondorState(
            expiry_time=pd.Timestamp(expiry_time),
            amount=amount,
            short_put=IronCondorLeg(short_put_name, short_put_strike, "PUT", "short"),
            long_put=IronCondorLeg(long_put_name, long_put_strike, "PUT", "long"),
            short_call=IronCondorLeg(short_call_name, short_call_strike, "CALL", "short"),
            long_call=IronCondorLeg(long_call_name, long_call_strike, "CALL", "long"),
        )

    def _close_condor(self, *, market: DeribitOptionMarket, state: IronCondorState) -> None:
        for action, instrument in (
            (market.sell, state.long_put.instrument_name),
            (market.sell, state.long_call.instrument_name),
            (market.buy, state.short_put.instrument_name),
            (market.buy, state.short_call.instrument_name),
        ):
            try:
                action(instrument, state.amount)
            except DemeterError as exc:
                logger.warning("Skipping close for %s: %s", str(instrument), str(exc))

    @staticmethod
    def _condor_positions_still_exist(*, market: DeribitOptionMarket, state: IronCondorState) -> bool:
        positions: dict[str, object] = market.positions
        needed: list[str] = [
            state.short_put.instrument_name,
            state.long_put.instrument_name,
            state.short_call.instrument_name,
            state.long_call.instrument_name,
        ]
        return all(name in positions for name in needed)

    def _select_expiry(self, *, orderbook_df: pd.DataFrame, now: pd.Timestamp) -> Optional[pd.Timestamp]:
        expiry_series: pd.Series = pd.to_datetime(orderbook_df["expiry_time"]).dropna().drop_duplicates().sort_values()
        if expiry_series.empty:
            return None

        min_expiry: pd.Timestamp = now + pd.Timedelta(days=self.min_dte_days)
        max_expiry: pd.Timestamp = now + pd.Timedelta(days=self.max_dte_days)

        in_band: pd.Series = expiry_series.loc[(expiry_series >= min_expiry) & (expiry_series <= max_expiry)]
        if not in_band.empty:
            return pd.Timestamp(in_band.iloc[0])

        future_only: pd.Series = expiry_series.loc[expiry_series > now]
        if not future_only.empty:
            return pd.Timestamp(future_only.iloc[0])

        return None

    @staticmethod
    def _instrument_for_strike(*, df: pd.DataFrame, strike_price: int) -> Optional[str]:
        strike_mask: pd.Series = df["strike_price"].astype(int) == int(strike_price)
        matched: pd.Index = df.index[strike_mask]
        if matched.empty:
            return None
        return str(matched[0])

    @staticmethod
    def _choose_put_strike_at_or_below(*, puts_df: pd.DataFrame, target: Decimal, spot: Decimal) -> Optional[int]:
        strikes: pd.Series = puts_df["strike_price"].dropna().astype(int).drop_duplicates()
        strikes = strikes.loc[strikes <= int(spot)]
        if strikes.empty:
            return None
        target_int: int = int(target)
        diffs: pd.Series = (strikes - target_int).abs()
        return int(strikes.iloc[int(diffs.argsort().iloc[0])])

    @staticmethod
    def _choose_put_strike_below(*, puts_df: pd.DataFrame, target: Decimal, below: Optional[int]) -> Optional[int]:
        if below is None:
            return None
        strikes: pd.Series = puts_df["strike_price"].dropna().astype(int).drop_duplicates()
        strikes = strikes.loc[strikes < int(below)]
        if strikes.empty:
            return None
        target_int: int = int(target)
        diffs: pd.Series = (strikes - target_int).abs()
        return int(strikes.iloc[int(diffs.argsort().iloc[0])])

    @staticmethod
    def _choose_call_strike_at_or_above(*, calls_df: pd.DataFrame, target: Decimal, spot: Decimal) -> Optional[int]:
        strikes: pd.Series = calls_df["strike_price"].dropna().astype(int).drop_duplicates()
        strikes = strikes.loc[strikes >= int(spot)]
        if strikes.empty:
            return None
        target_int: int = int(target)
        diffs: pd.Series = (strikes - target_int).abs()
        return int(strikes.iloc[int(diffs.argsort().iloc[0])])

    @staticmethod
    def _choose_call_strike_above(*, calls_df: pd.DataFrame, target: Decimal, above: Optional[int]) -> Optional[int]:
        if above is None:
            return None
        strikes: pd.Series = calls_df["strike_price"].dropna().astype(int).drop_duplicates()
        strikes = strikes.loc[strikes > int(above)]
        if strikes.empty:
            return None
        target_int: int = int(target)
        diffs: pd.Series = (strikes - target_int).abs()
        return int(strikes.iloc[int(diffs.argsort().iloc[0])])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    market: DeribitOptionMarket = DeribitOptionMarket(market_key, DeribitOptionMarket.ETH)

    # SAME loading path as your working example
    start_day: date = date(2024, 5, 11)
    end_day: date = date(2024, 7, 20)
    data_path = Path(__file__).resolve().parents[2] / "tests" / "data" / "eth_db"
    underlying: str = "ETH"

    _ = validate_deribit_option_book_files(
        data_path=data_path,
        start_date=start_day,
        end_date=end_day,
        underlying=underlying,
        sample_rows_per_file=500,
        strict_numeric=False,  # ADD
        max_coerced_ratio=0.10,  # CHANGE (was 0.01)
    )


    market: DeribitOptionMarket = DeribitOptionMarket(market_key, DeribitOptionMarket.ETH)
    data: pd.DataFrame = load_deribit_option_data(start_day, end_day, data_path=str(data_path))

    # ADD: post-load numeric hardening so downstream caching/pyarrow paths cannot choke on object dtype
    data = _coerce_numeric_columns_for_arrow(data, max_coerced_ratio=0.10)

    market.data = data

    actuator: Actuator = Actuator()
    actuator.broker.add_market(market)

    actuator.broker.set_balance(DeribitOptionMarket.ETH, 10)
    market.deposit(10)

    actuator.strategy = RollingIronCondorStrategy(
        amount_multiplier=Decimal("1"),
        short_distance_pct=Decimal("0.05"),
        wing_distance_pct=Decimal("0.10"),
        min_dte_days=7,
        max_dte_days=35,
    )

    actuator.set_price(get_price_from_data(data))
    actuator.run()
    actuator.save_result(path="./result", file_name="deribit_rolling_iron_condor", decimals=3)
