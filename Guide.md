# Extract SXXP data with partial mode (last 500 days)
python bloomberg_data_extraction.py sxxp

# Extract SPX data with full mode (10 years)
python bloomberg_data_extraction.py spx --mode full

# Extract risk proxies only
python bloomberg_data_extraction.py risk_proxies

# Extract all data (SXXP, SPX, risk proxies, and indices)
python bloomberg_data_extraction.py all

# Use custom base directory
python bloomberg_data_extraction.py all --base-dir "C:\MyData"