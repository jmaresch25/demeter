import logging
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd

from demeter import Actuator, MarketInfo, Snapshot, Strategy, TokenInfo
from demeter.deribit.market import DeribitOptionMarket  # provides buy(), sell(), get_market_balance(), market_status


# -----------------------------
# Core idea (in 2 lines)
# -----------------------------
# 1) Open a SHORT iron condor (credit): sell OTM put/call + buy further OTM wings.
# 2) If spot "touches" either short strike, close all 4 legs and open a new condor around the new spot.


# -----------------------------
# IMPORTANT ASSUMPTION
# -----------------------------
# This strategy assumes DeribitOptionMarket supports:
# - sell-to-open (creating short option positions) via market.sell(...)
# - buy-to-close shorts via market.buy(...)
# If your implementation only supports sell-to-close (i.e., cannot create short positions),
# you must extend the market model before a true iron condor can be simulated.


@dataclass(frozen=True)
class IronCondorLeg:
    instrument_name: str
    strike_price: int
    option_type: str  # "PUT" or "CALL"
    side: str  # "short" or "long"


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
        market_key: MarketInfo,
        token: TokenInfo,
        amount_multiplier: Decimal = Decimal("1"),
        short_distance_pct: Decimal = Decimal("0.05"),  # short strikes ~ 5% away from spot
        wing_distance_pct: Decimal = Decimal("0.10"),  # long wings ~ 10% away from spot
        min_dte_days: int = 7,
        max_dte_days: int = 35,
    ):
        super().__init__()
        self.market_key: MarketInfo = market_key
        self.token: TokenInfo = token

        self.amount_multiplier: Decimal = amount_multiplier
        self.short_distance_pct: Decimal = short_distance_pct
        self.wing_distance_pct: Decimal = wing_distance_pct
        self.min_dte_days: int = min_dte_days
        self.max_dte_days: int = max_dte_days

        self.condor_state: Optional[IronCondorState] = None

    def on_bar(self, row_data: Snapshot) -> None:
        market: DeribitOptionMarket = self.markets[self.market_key]

        # DeribitOptionMarket is only writable on exact hourly timestamps (per its implementation).
        is_open_bar: bool = market.market_status.timestamp == market.market_status.timestamp.floor("1h")
        if not is_open_bar:
            return

        # If no orderbook slice exists at this hour, skip.
        orderbook_df: pd.DataFrame = market.market_status.data
        if orderbook_df is None or orderbook_df.empty:
            return

        spot_usd: Decimal = Decimal(str(orderbook_df.iloc[0]["underlying_price"]))

        # If we have no condor open, open one.
        if self.condor_state is None:
            self._open_new_condor(market=market, orderbook_df=orderbook_df, now=row_data.timestamp, spot_usd=spot_usd)
            return

        # If any leg is missing from positions (e.g., expired/exercised by market.update()), reset state.
        if not self._condor_positions_still_exist(market=market, state=self.condor_state):
            self.condor_state = None
            return

        # Roll trigger: spot touches either short strike.
        short_put_k: Decimal = Decimal(self.condor_state.short_put.strike_price)
        short_call_k: Decimal = Decimal(self.condor_state.short_call.strike_price)
        touched_short_put: bool = spot_usd <= short_put_k
        touched_short_call: bool = spot_usd >= short_call_k
        if not (touched_short_put or touched_short_call):
            return

        # Roll: close then reopen around current spot.
        self._close_condor(market=market, state=self.condor_state)
        self.condor_state = None
        self._open_new_condor(market=market, orderbook_df=orderbook_df, now=row_data.timestamp, spot_usd=spot_usd)

    def _open_new_condor(
        self,
        market: DeribitOptionMarket,
        orderbook_df: pd.DataFrame,
        now: pd.Timestamp,
        spot_usd: Decimal,
    ) -> None:
        # Pick an expiry in a target DTE band (fallback to nearest future expiry if band is empty).
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

        # Targets for strikes (in USD).
        sp_target: Decimal = spot_usd * (Decimal("1") - self.short_distance_pct)
        lp_target: Decimal = spot_usd * (Decimal("1") - self.wing_distance_pct)
        sc_target: Decimal = spot_usd * (Decimal("1") + self.short_distance_pct)
        lc_target: Decimal = spot_usd * (Decimal("1") + self.wing_distance_pct)

        # Choose strikes with structural constraints:
        # - PUT side: long_put_strike < short_put_strike <= spot
        # - CALL side: spot <= short_call_strike < long_call_strike
        short_put_strike: Optional[int] = self._choose_put_strike_at_or_below(puts_df=puts_df, target=sp_target, spot=spot_usd)
        long_put_strike: Optional[int] = self._choose_put_strike_below(puts_df=puts_df, target=lp_target, below=short_put_strike)
        short_call_strike: Optional[int] = self._choose_call_strike_at_or_above(calls_df=calls_df, target=sc_target, spot=spot_usd)
        long_call_strike: Optional[int] = self._choose_call_strike_above(calls_df=calls_df, target=lc_target, above=short_call_strike)

        if (
            short_put_strike is None
            or long_put_strike is None
            or short_call_strike is None
            or long_call_strike is None
        ):
            return

        short_put_name: str = str(puts_df.loc[puts_df["strike_price"] == short_put_strike].index[0])
        long_put_name: str = str(puts_df.loc[puts_df["strike_price"] == long_put_strike].index[0])
        short_call_name: str = str(calls_df.loc[calls_df["strike_price"] == short_call_strike].index[0])
        long_call_name: str = str(calls_df.loc[calls_df["strike_price"] == long_call_strike].index[0])

        # Contract amount: use market dust/min amount * multiplier (kept simple and deterministic).
        base_amount: Decimal = market.token_config.min_amount
        amount: Decimal = base_amount * self.amount_multiplier

        # Open SHORT iron condor in a cash-friendly order:
        # 1) sell shorts (receive premium), 2) buy wings (pay premium)
        market.sell(short_put_name, amount)
        market.sell(short_call_name, amount)
        market.buy(long_put_name, amount)
        market.buy(long_call_name, amount)

        self.condor_state = IronCondorState(
            expiry_time=expiry_time,
            amount=amount,
            short_put=IronCondorLeg(short_put_name, short_put_strike, "PUT", "short"),
            long_put=IronCondorLeg(long_put_name, long_put_strike, "PUT", "long"),
            short_call=IronCondorLeg(short_call_name, short_call_strike, "CALL", "short"),
            long_call=IronCondorLeg(long_call_name, long_call_strike, "CALL", "long"),
        )

    def _close_condor(self, market: DeribitOptionMarket, state: IronCondorState) -> None:
        # Close in a cash-friendly order:
        # - Sell long wings first (receive premium), then buy back short legs.
        market.sell(state.long_put.instrument_name, state.amount)
        market.sell(state.long_call.instrument_name, state.amount)
        market.buy(state.short_put.instrument_name, state.amount)
        market.buy(state.short_call.instrument_name, state.amount)

    @staticmethod
    def _condor_positions_still_exist(market: DeribitOptionMarket, state: IronCondorState) -> bool:
        positions: Dict[str, object] = market.positions
        needed: List[str] = [
            state.short_put.instrument_name,
            state.long_put.instrument_name,
            state.short_call.instrument_name,
            state.long_call.instrument_name,
        ]
        return all(name in positions for name in needed)

    def _select_expiry(self, orderbook_df: pd.DataFrame, now: pd.Timestamp) -> Optional[pd.Timestamp]:
        expiry_series: pd.Series = pd.to_datetime(orderbook_df["expiry_time"]).dropna().drop_duplicates().sort_values()
        if expiry_series.empty:
            return None

        now_ts: pd.Timestamp = pd.Timestamp(now)
        min_expiry: pd.Timestamp = now_ts + pd.Timedelta(days=self.min_dte_days)
        max_expiry: pd.Timestamp = now_ts + pd.Timedelta(days=self.max_dte_days)

        in_band: pd.Series = expiry_series.loc[(expiry_series >= min_expiry) & (expiry_series <= max_expiry)]
        if not in_band.empty:
            return pd.Timestamp(in_band.iloc[0])

        future_only: pd.Series = expiry_series.loc[expiry_series > now_ts]
        if not future_only.empty:
            return pd.Timestamp(future_only.iloc[0])

        return None

    @staticmethod
    def _choose_put_strike_at_or_below(puts_df: pd.DataFrame, target: Decimal, spot: Decimal) -> Optional[int]:
        strikes: pd.Series = puts_df["strike_price"].dropna().astype(int).drop_duplicates()
        strikes = strikes.loc[strikes <= int(spot)]
        if strikes.empty:
            return None
        target_int: int = int(target)
        chosen: int = int(strikes.iloc[(strikes - target_int).abs().argsort().iloc[0]])
        return chosen

    @staticmethod
    def _choose_put_strike_below(puts_df: pd.DataFrame, target: Decimal, below: Optional[int]) -> Optional[int]:
        if below is None:
            return None
        strikes: pd.Series = puts_df["strike_price"].dropna().astype(int).drop_duplicates()
        strikes = strikes.loc[strikes < int(below)]
        if strikes.empty:
            return None
        target_int: int = int(target)
        chosen: int = int(strikes.iloc[(strikes - target_int).abs().argsort().iloc[0]])
        return chosen

    @staticmethod
    def _choose_call_strike_at_or_above(calls_df: pd.DataFrame, target: Decimal, spot: Decimal) -> Optional[int]:
        strikes: pd.Series = calls_df["strike_price"].dropna().astype(int).drop_duplicates()
        strikes = strikes.loc[strikes >= int(spot)]
        if strikes.empty:
            return None
        target_int: int = int(target)
        chosen: int = int(strikes.iloc[(strikes - target_int).abs().argsort().iloc[0]])
        return chosen

    @staticmethod
    def _choose_call_strike_above(calls_df: pd.DataFrame, target: Decimal, above: Optional[int]) -> Optional[int]:
        if above is None:
            return None
        strikes: pd.Series = calls_df["strike_price"].dropna().astype(int).drop_duplicates()
        strikes = strikes.loc[strikes > int(above)]
        if strikes.empty:
            return None
        target_int: int = int(target)
        chosen: int = int(strikes.iloc[(strikes - target_int).abs().argsort().iloc[0]])
        return chosen


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    # Market setup
    token: TokenInfo = TokenInfo("ETH", 18)  # Demeter expects token names in upper case for price columns.
    market_key: MarketInfo = MarketInfo("deribit_eth_options")
    data_path = Path(__file__).resolve().parents[2] / "tests" / "data"
    market: DeribitOptionMarket = DeribitOptionMarket(market_info=market_key, token=token, data_path=str(data_path))

    # Data window (must match your downloaded Deribit-option-book-ETH-YYYYMMDD.csv files)
    start: date = date(2024, 1, 1)
    end: date = date(2024, 1, 31)
    market.load_data(start_date=start, end_date=end)

    # Build price series from the option data (hourly -> BASE_FREQ)
    price_df: pd.DataFrame = market.get_price_from_data()
    if "ETH" not in price_df.columns:
        # Defensive rename if your helper produced a different column label.
        price_df.columns = ["ETH"]

    # Actuator / Broker wiring
    actuator: Actuator = Actuator()
    actuator.set_price(price_df)
    actuator.broker.add_market(market)

    # Fund the broker wallet, then deposit into the Deribit market sub-wallet.
    initial_wallet_eth: Decimal = Decimal("200")
    deposit_to_deribit_eth: Decimal = Decimal("150")
    actuator.broker.set_balance(token, initial_wallet_eth)
    market.deposit(deposit_to_deribit_eth)

    # Strategy
    actuator.strategy = RollingIronCondorStrategy(
        market_key=market_key,
        token=token,
        amount_multiplier=Decimal("1"),
        short_distance_pct=Decimal("0.05"),
        wing_distance_pct=Decimal("0.10"),
        min_dte_days=7,
        max_dte_days=35,
    )

    # Run backtest
    actuator.run()

