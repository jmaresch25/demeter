from datetime import datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from demeter import MarketInfo, MarketTypeEnum
from demeter.deribit import DeribitOptionMarket
from demeter.deribit._typing import DeribitMarketStatus, OptionPosition, OptionKind
from demeter._typing import DemeterError


def _build_orderbook_df(
    timestamp: pd.Timestamp,
    instrument_name: str,
    *,
    underlying_price: float = 2100.0,
    mark_price: float = 0.05,
) -> pd.DataFrame:
    expiry_time = timestamp + timedelta(days=14)
    data = {
        "state": "open",
        "type": "PUT",
        "strike_price": 2000,
        "t": 0.0,
        "expiry_time": expiry_time,
        "vega": 0.0,
        "theta": 0.0,
        "rho": 0.0,
        "gamma": 0.0,
        "delta": 0.0,
        "underlying_price": underlying_price,
        "settlement_price": None,
        "mark_price": mark_price,
        "mark_iv": 0.0,
        "last_price": 0.0,
        "interest_rate": 0.0,
        "bid_iv": 0.0,
        "best_bid_price": 0.05,
        "best_bid_amount": 5.0,
        "ask_iv": 0.0,
        "best_ask_price": 0.06,
        "best_ask_amount": 5.0,
        "asks": [[0.06, 10.0]],
        "bids": [[0.05, 10.0]],
    }
    index = pd.MultiIndex.from_tuples([(timestamp, instrument_name)], names=["time", "instrument_name"])
    return pd.DataFrame([data], index=index)


def _set_market_status(market: DeribitOptionMarket, timestamp: pd.Timestamp, *, price_value: float = 2100.0) -> None:
    price = pd.Series({"ETH": price_value}, name=timestamp)
    market.set_market_status(DeribitMarketStatus(timestamp=timestamp, data=None), price)


def test_sell_to_open_and_buy_to_close_short_position():
    market_key = MarketInfo("deribit_test", MarketTypeEnum.deribit_option)
    market = DeribitOptionMarket(market_key, DeribitOptionMarket.ETH)
    timestamp = pd.Timestamp(datetime(2024, 2, 15, 8, 0, 0))
    instrument_name = "ETH-15FEB24-2000-P"
    market.data = _build_orderbook_df(timestamp, instrument_name, underlying_price=1500.0)
    _set_market_status(market, timestamp, price_value=1500.0)
    market._add_to_balance(Decimal("10"))

    market.sell(instrument_name, Decimal("1"))
    assert instrument_name in market.positions
    assert market.positions[instrument_name].amount == Decimal("-1")

    market.buy(instrument_name, Decimal("1"))
    assert instrument_name not in market.positions

    market.sell(instrument_name, Decimal("1"))
    with pytest.raises(DemeterError):
        market.buy(instrument_name, Decimal("2"))


def test_short_position_settlement_debits_balance():
    market_key = MarketInfo("deribit_test", MarketTypeEnum.deribit_option)
    market = DeribitOptionMarket(market_key, DeribitOptionMarket.ETH)
    timestamp = pd.Timestamp(datetime(2024, 2, 15, 8, 0, 0))
    instrument_name = "ETH-15FEB24-2000-P"
    market.data = _build_orderbook_df(timestamp, instrument_name, underlying_price=1500.0)
    _set_market_status(market, timestamp, price_value=1500.0)
    market._add_to_balance(Decimal("10"))

    position = OptionPosition(
        instrument_name=instrument_name,
        expiry_time=timestamp,
        strike_price=2000,
        type=OptionKind.put,
        amount=Decimal("-1"),
        avg_buy_price=Decimal("0"),
        buy_amount=Decimal("0"),
        avg_sell_price=Decimal("0.05"),
        sell_amount=Decimal("1"),
    )
    market.positions[instrument_name] = position
    balance_before = market.balance
    market.check_option_exercise()

    assert market.balance < balance_before
    assert instrument_name not in market.positions
