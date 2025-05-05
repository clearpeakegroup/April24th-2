from ironbeam import IronbeamREST, IronbeamWS
import os

class IronbeamBroker:
    def __init__(self):
        self.api_key = os.environ.get('IRONBEAM_API_KEY')
        self.api_secret = os.environ.get('IRONBEAM_API_SECRET')
        self.account = os.environ.get('IRONBEAM_ACCOUNT')
        self.account_password = os.environ.get('IRONBEAM_ACCOUNT_PASSWORD')
        if not all([self.api_key, self.api_secret, self.account, self.account_password]):
            raise RuntimeError("Ironbeam credentials must be set in environment variables: IRONBEAM_API_KEY, IRONBEAM_API_SECRET, IRONBEAM_ACCOUNT, IRONBEAM_ACCOUNT_PASSWORD")
        self.rest = IronbeamREST(
            api_key=self.api_key,
            api_secret=self.api_secret,
            account=self.account,
            password=self.account_password
        )
        self.ws = IronbeamWS(
            api_key=self.api_key,
            api_secret=self.api_secret,
            account=self.account,
            password=self.account_password
        )

    def cancel_order(self, order_id, **kwargs):
        return self.rest.cancel_order(order_id)

    def get_order_status(self, order_id, **kwargs):
        return self.rest.get_order_status(order_id)

    def get_account(self, **kwargs):
        return self.rest.get_account()

    # Add more methods as needed for place_order, etc. 