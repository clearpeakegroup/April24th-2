#!/bin/bash
# Store secrets in Vault
vault kv put secret/finrl/databento api_key=YOUR_DATABENTO_KEY
vault kv put secret/finrl/etrade api_key=YOUR_ETRADE_KEY api_secret=YOUR_ETRADE_SECRET
vault kv put secret/finrl/db user=finrl password=finrlpass host=localhost port=5433 dbname=finrl_db 