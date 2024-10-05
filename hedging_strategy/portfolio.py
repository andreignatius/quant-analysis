import polars as pl
from typing import Optional

class portfolio:
    def __init__(
        self,
        hedge_date: str,
        holding_value: pl.DataFrame,
        
    ) -> None:
        
        self.hedge_date = hedge_date
        self.holding_value = holding_value
    
    def compute_portfolio(
        self,
        hedge_value: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        holding = (
            self.holding_value
            .filter(
                pl.col('Date')
                >= pl.lit(self.hedge_date).str.strptime(pl.Date, '%Y-%m-%d')
            )
        )

        if hedge_value is not None:
            portfolio = (
                hedge_value
                .join(
                    holding,
                    how = 'left',
                    left_on = 'date',
                    right_on = 'Date'
                )
                .with_columns(
                    (pl.col('hedge_value') + pl.col('Holding')).alias('port_value')
                )
                .with_columns(
                    pl.col('port_value')
                    .pct_change()
                    .alias('returns')
                )
            )
        else:
            portfolio = (
                holding
                .clone()
                .rename(
                    {
                        'Holding' : 'port_value'
                    }
                )
                .with_columns(
                    pl.col('port_value')
                    .pct_change()
                    .alias('returns')
                )
            )
        
        return portfolio