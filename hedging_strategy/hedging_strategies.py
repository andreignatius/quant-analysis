import polars as pl


class hedging:
    def __init__(
        self,
        hedge_date,
        options,
        close
    ) -> None:
        
        self.hedge_date = hedge_date
        self.options = options
        self.close = close
        
    def buy_put(
        self
    ) -> pl.DataFrame:
        day_option_chain = (
            self.options
            .filter(
                pl.col('date') == pl.lit(self.hedge_date).str.strptime(pl.Date, '%Y-%m-%d'),
                pl.col('days_to_expiry') > 365,
                pl.col('cp_flag') == 'P'
            )
        )
        
        strike = (
            self.close 
            - 
            min(
                abs(
                    day_option_chain
                    ['strike']
                    .unique() 
                    - self.close
                )
            )
        )
        
        symbol = (
            day_option_chain
            .filter(
                pl.col('strike') == strike
            )
            .select(
                'symbol'
            )
            .item()
        )
        
        long_put = (
            self.options
            .filter(
                pl.col('date') >= pl.lit(self.hedge_date).str.strptime(pl.Date, '%Y-%m-%d'),
                pl.col('symbol') == symbol
            )
        )
        
        daily_value = (
            long_put
            .select(
                'date', 'best_bid'
            )
            .with_columns(
                pl.col('best_bid') * 100
            )
            .rename(
                {
                    'best_bid' : 'hedge_value'
                }
            )
        )
                
        return daily_value
    
    def short_call(
        self
    ) -> pl.DataFrame:
        day_option_chain = (
            self.options
            .filter(
                pl.col('date') == pl.lit(self.hedge_date).str.strptime(pl.Date, '%Y-%m-%d'),
                pl.col('days_to_expiry') > 365,
                pl.col('cp_flag') == 'C'
            )
        )
        
        strike = (
            self.close 
            - 
            min(
                abs(
                    day_option_chain
                    ['strike']
                    .unique() 
                    - self.close
                )
            )
        )
        
        symbol = (
            day_option_chain
            .filter(
                pl.col('strike') == strike
            )
            .select(
                'symbol'
            )
            .item()
        )
        
        short_call = (
            self.options
            .filter(
                pl.col('date') >= pl.lit(self.hedge_date).str.strptime(pl.Date, '%Y-%m-%d'),
                pl.col('symbol') == symbol
            )
        )
        
        daily_value = (
            short_call
            .select(
                'date', 'best_offer'
            )
            .with_columns(
                -pl.col('best_offer') * 100
            )
            .rename(
                {
                    'best_offer' : 'hedge_value'
                }
            )
        )
        
        return daily_value