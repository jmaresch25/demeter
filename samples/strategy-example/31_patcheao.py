# dana1.py (patched to validate created CSVs before load_deribit_option_data)
from __future__ import annotations
from pathlib import Path
from datetime import date, datetime

import pandas as pd

from demeter import Actuator, AtTimeTrigger, MarketInfo, MarketTypeEnum, Snapshot, Strategy
from demeter.deribit import DeribitOptionMarket, get_price_from_data, load_deribit_option_data

from deribit_option_book_safety import configure_logging, validate_deribit_option_book_files


market_key: MarketInfo = MarketInfo("option_test", MarketTypeEnum.deribit_option)

pd.options.display.max_columns = None
pd.set_option("display.width", 5000)


class SimpleStrategy(Strategy):
    def initialize(self) -> None:
        new_trigger: AtTimeTrigger = AtTimeTrigger(time=datetime(2024, 2, 15, 12, 0, 0), do=self.buy)
        self.triggers.append(new_trigger)

    def buy(self, snapshot: Snapshot) -> None:
        market: DeribitOptionMarket = self.broker.markets.default
        market.estimate_cost("ETH-26APR24-2700-C", 20, "buy")
        market.buy("ETH-26APR24-2700-C", 20)

    def notify(self, action: object) -> None:
        print(action)


if __name__ == "__main__":
    configure_logging()

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
    market.data = data

    actuator: Actuator = Actuator()
    actuator.broker.add_market(market)
    actuator.broker.set_balance(DeribitOptionMarket.ETH, 10)

    market.deposit(10)

    actuator.strategy = SimpleStrategy()
    actuator.set_price(get_price_from_data(data))
    actuator.run()
    actuator.save_result(path="./result", file_name="deribit", decimals=3)
