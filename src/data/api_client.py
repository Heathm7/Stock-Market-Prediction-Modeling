from src.data.providers.alpha_vantage import AlphaVantageProvider

class MarketAPIClient:
    def __init__(self, provider: str, **kwargs):
        if provider == "alpha_vantage":
            self.provider = AlphaVantageProvider(**kwargs)
        #   elif provider == "finnhub":
        #       self.provider = FinnhubProvider(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def fetch_historical_data(self, *args, **kwargs):
        return self.provider.fetch_historical_data(*args, **kwargs)




