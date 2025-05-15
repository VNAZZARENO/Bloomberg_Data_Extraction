# Bloomberg Data Extraction Guide

## Basic Usage

### Extract SXXP data with partial mode (last 100 days)
```
python bloomberg_data_extraction.py sxxp
```

### Extract SPX data with full mode (15 years)
```
python bloomberg_data_extraction.py spx --mode full
```

### Extract risk proxies only
```
python bloomberg_data_extraction.py risk_proxies
```

### Extract all data (SXXP, SPX, risk proxies, and indices)
```
python bloomberg_data_extraction.py all
```

## Options

### Custom base directory
```
python bloomberg_data_extraction.py all --base-dir "C:\MyData"
```

### Currency Conversion
By default, the script converts all stock prices to EUR based on the market suffix (e.g., 'FP' for Paris, 'LN' for London). This creates normalized prices for cross-market analysis.

#### Disable currency conversion
```
python bloomberg_data_extraction.py all --no-currency-conversion
```

When currency conversion is enabled:
- Exchange rates are fetched from Bloomberg and saved in `data/fx_rates/exchange_rates.csv`
- Original prices are preserved in `*_original.csv` files
- Converted prices (normalized to EUR) are saved in the standard output files

## Market Suffix to Currency Mapping

The script uses the following mapping to determine which currency to convert from:

- **Eurozone (EUR)**: FP (Paris), GY (Germany), IM (Italy), SM (Spain), etc.
- **Non-Eurozone European**: LN (London, GBP), SS (Sweden, SEK), SW (Switzerland, CHF), etc.
- **Americas**: US/UN/UW (United States, USD), CN (Canada, CAD), etc.
- **Asia-Pacific**: JT (Japan, JPY), HK (Hong Kong, HKD), AU (Australia, AUD), etc.