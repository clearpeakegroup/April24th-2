## EdgeClear / Ironbeam Integration

Add the following to your `.env`:

```
IRONBEAM_CLIENT_ID=         # Your Ironbeam OAuth2 client ID
IRONBEAM_CLIENT_SECRET=     # Your Ironbeam OAuth2 client secret
IRONBEAM_SANDBOX=true       # Use Ironbeam demo/sandbox environment
IRONBEAM_BASE_URL=https://api.ironbeam.com/v2
IRONBEAM_WS_URL=wss://md.ironbeam.com/v2/stream
SYMBOLS=CME.MES*,CME.MNQ*,CME.ZF*,CME.ZN*,CME.UB*
NVCOMP_ZSTD=1               # Enable GPU ZSTD decompression via nvCOMP
```

- `IRONBEAM_SANDBOX` toggles between demo and live trading.
- `SYMBOLS` is a comma-separated list of all supported symbols for the five-feed data layer.
- `NVCOMP_ZSTD` enables GPU-accelerated decompression for historical and live data. 