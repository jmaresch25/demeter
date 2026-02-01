
import logging
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from demeter import Strategy, MarketInfo, Actuator, Snapshot, MarketTypeEnum
from demeter.deribit import DeribitOptionMarket, load_deribit_option_data, get_price_from_data


logger: logging.Logger = logging.getLogger(__name__)

pd.options.display.max_columns = None
pd.set_option("display.width", 5000)

market_key: MarketInfo = MarketInfo("option_test", MarketTypeEnum.deribit_option)


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
        # No AtTimeTrigger: this strategy continuously evaluates roll conditions.
        return

    def on_bar(self, snapshot: Snapshot) -> None:
        market: DeribitOptionMarket = self.broker.markets.default

        # DeribitOptionMarket is writable only on exact-hour timestamps (per its implementation).
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

        # Open SHORT iron condor (credit)
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
        # Close: sell wings first, then buy back shorts
        market.sell(state.long_put.instrument_name, state.amount)
        market.sell(state.long_call.instrument_name, state.amount)
        market.buy(state.short_put.instrument_name, state.amount)
        market.buy(state.short_call.instrument_name, state.amount)

    @staticmethod
    def _condor_positions_still_exist(*, market: DeribitOptionMarket, state: IronCondorState) -> bool:
        positions: Dict[str, object] = market.positions
        needed: List[str] = [
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
    data_path: Path = Path(__file__).resolve().parents[2] / "tests" / "data"
    data: pd.DataFrame = load_deribit_option_data(date(2024, 2, 15), date(2024, 2, 16), data_path=str(data_path))
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
