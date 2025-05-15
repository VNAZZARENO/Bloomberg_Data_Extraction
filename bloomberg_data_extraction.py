#!/usr/bin/env python
from xbbg import blp
import pandas as pd
import datetime as dt
from datetime import timedelta
import os
import argparse
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BloombergDataExtractor:
    def __init__(self, base_dir, extraction_mode='partial'):
        self.base_dir = base_dir
        self.output_dir = os.path.join(base_dir, "data")
        self.extraction_mode = extraction_mode
        self.today = dt.datetime.today()
        if extraction_mode == 'full':
            self.start_date = self.today - timedelta(days=15 * 365)  
        else:
            self.start_date = self.today - timedelta(days=100)  
        self.end_date = self.today + timedelta(days=1)
        
        self.field_mappings = {
            'PX_LAST': 'price.csv',
            'CALL_IMP_VOL_10D': 'vol_10d.csv',
            'CALL_IMP_VOL_30D': 'vol_1m.csv',
            'CALL_IMP_VOL_60D': 'vol_2m.csv',
            'VOLUME': 'volume.csv',
            'EARNINGS_RELATED_IMPLIED_MOVE': 'earning_implied_move.csv'
        }
        
        self.index_tickers = ['SXXP Index', 'SXPP Index', 'SXFP Index', 'SX7P Index', 
                            'SXDP Index', 'SXAP Index', 'SX6P Index', 'SXIP Index', 
                            'SXNP Index', 'SX5E Index', 'SPX Index', 'SPW Index', 'WLSNRE Index']
        
        self.risk_proxy_tickers = [
            'PERGLFI FP Equity', 'H15T3M Index', 'H15T1Y Index', 'H15T5Y Index', 
            'H15T10Y Index', 'ESES Index', 'OATA Comdty', 'VGA Index', 
            'EURUSD Curncy', 'NZDUSD Curncy', 'AUDUSD Curncy', 'EURGBP Curncy', 
            'CHFUSD Curncy', 'CHFEUR Curncy', 'USDJPY Curncy', 'EURJPY Curncy', 
            'EURCHF Index', 'XBTUSD Curncy', 'XAU Curncy', 'ERA Comdty', 
            'FRANCE CDS USD SR 10Y D14 Corp', 'UK CDS USD SR 10Y D14 Corp', 
            'SPAIN CDS USD SR 10Y D14 Corp', 'ITALY CDS USD SR 10Y D14 Corp', 
            'GERMAN CDS USD SR 10Y D14 Corp', 'GTDEM10Y Govt', 'CTDEM10Y Govt', 
            'GDP CURY Index', 'CTFRF10Y Govt', 'CTITL10Y Govt', 
            'CTEUR10Y Govt', 'GUKG30 Index', 'CTEUR7Y Govt', 'JPEI3MEU Index', 
            'LF98TRUU Index', 'IBXXCHF3 Index', 'CLA Comdty', 'ASDA Index', 
            'DEDA Index', 'VIX Index', 'VDAX Index', 'VCAC Index', 'Move Index', 
            'V2X Index', 'BCOM Index', 'VG1 Index', 'ITRX EUR CDSI GEN 5Y Corp', 
            'ITRX XOVER CDSI GEN 5Y Corp', 'ITRX JAPAN CDSI GEN 5Y Corp', 
            'ITRX EXJP IG CDSI GEN 5Y Corp', 'HGK5 Index', 'SIK5 Index', 
            'XPT BGN Curncy', 'ITRX EXJP IG CDSI S42 5Y Corp', 'USYC2Y10 Index', 
            'CDX IG CDSI GEN 5Y Corp', 'CDX HY CDSI GEN 5Y Corp', 'SPX Index', 
            'SPW Index', 'WLSNRE Index'
        ]
        
        self._remove_duplicates()
        self.index_to_rate_mapping = {
            'SXXP': 'GTDEM10Y Govt',
            'SPX': 'USGG10YR Index',
            'SPW': 'USGG10YR Index',
            'SX5E': 'GTDEM10Y Govt',
            'FTSE': 'GUKG10 Index',
        }
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _remove_duplicates(self):
        seen = set()
        unique_risk_tickers = []
        for ticker in self.risk_proxy_tickers:
            if ticker not in seen:
                seen.add(ticker)
                unique_risk_tickers.append(ticker)
        self.risk_proxy_tickers = unique_risk_tickers
        duplicates = len(self.risk_proxy_tickers) - len(unique_risk_tickers)
        if duplicates > 0:
            logger.info(f"Suppression de {duplicates} doublons dans risk_proxy_tickers")
    
    def extract_data_for_universe(self, tickers, fields=['PX_LAST']):
        try:
            unique_tickers = list(dict.fromkeys(tickers))
            if len(unique_tickers) < len(tickers):
                logger.warning(f"Doublons détectés et supprimés. {len(tickers)} -> {len(unique_tickers)} tickers")
                tickers = unique_tickers
            
            logger.info(f"Extracting data for {len(tickers)} tickers from {self.start_date} to {self.end_date}")
            df = blp.bdh(tickers=tickers, flds=fields, start_date=self.start_date, end_date=self.end_date)
            logger.info(f"Successfully extracted data for {len(tickers)} tickers")
            return df
        except Exception as e:
            logger.error(f"Failed to extract data: {e}")
            logger.error(f"Tickers causing issue: {tickers}")
            return pd.DataFrame()
    
    def merge_with_existing_data(self, new_df, file_path):
        if os.path.exists(file_path):
            try:
                existing_df = pd.read_csv(file_path, index_col=0, sep=';')
                logger.info(f"Existing data shape: {existing_df.shape}")
                logger.info(f"Existing data date range: {existing_df.index[0]} to {existing_df.index[-1]}")
                existing_df.index.name = 'Dates'
                try:
                    existing_df.index = pd.to_datetime(existing_df.index, format='%d/%m/%Y')
                except:
                    try:
                        existing_df.index = pd.to_datetime(existing_df.index, dayfirst=True)
                    except:
                        existing_df.index = pd.to_datetime(existing_df.index, errors='coerce')
                if not isinstance(new_df.index, pd.DatetimeIndex):
                    new_df.index = pd.to_datetime(new_df.index)
                new_df.index.name = 'Dates'
                
                logger.info(f"New data shape: {new_df.shape}")
                logger.info(f"New data date range: {new_df.index[0]} to {new_df.index[-1]}")
                existing_cols = list(existing_df.columns)
                new_cols = list(new_df.columns)
                
                existing_df.columns = [col.strip() for col in existing_df.columns]
                new_df.columns = [col.strip() for col in new_df.columns]
                
                if set(existing_cols) != set(new_cols):
                    logger.warning(f"Column mismatch - Existing: {existing_cols}, New: {new_cols}")
                    common_cols = list(set(existing_cols) & set(new_cols))
                    if common_cols:
                        existing_df = existing_df[common_cols]
                        new_df = new_df[common_cols]
                    else:
                        logger.error("No common columns found between existing and new data")
                        return new_df
                
                combined_df = existing_df.copy()
                combined_df.update(new_df)
                new_dates = new_df.index.difference(existing_df.index)
                if len(new_dates) > 0:
                    combined_df = pd.concat([combined_df, new_df.loc[new_dates]])
                
                combined_df = combined_df.sort_index()
                logger.info(f"Combined data shape: {combined_df.shape}")
                logger.info(f"Combined data date range: {combined_df.index[0]} to {combined_df.index[-1]}")
                
                return combined_df
            except Exception as e:
                logger.warning(f"Failed to merge with existing data: {e}")
                logger.warning(f"Detailed error: {type(e).__name__}: {str(e)}")
                backup_path = file_path.replace('.csv', '_backup.csv')
                logger.warning(f"Creating backup at {backup_path}")
                new_df.to_csv(backup_path, sep=';', date_format='%d/%m/%Y')
                new_df.index.name = 'Dates'
                return new_df
        else:
            new_df.index.name = 'Dates'
            return new_df
    
    def process_and_save_data(self, df, tickers, save_dir, field_mappings=None):
        """Process the dataframe and save to appropriate CSV files avec format de date cohérent"""
        if field_mappings is None:
            field_mappings = {'PX_LAST': 'price.csv'}
        
        os.makedirs(save_dir, exist_ok=True)
        
        if df.empty:
            logger.warning("Empty dataframe, skipping save")
            return
        
        df.index.name = 'Dates'
        if isinstance(df.columns, pd.MultiIndex):
            for field in field_mappings.keys():
                try:
                    field_data = df.xs(field, axis=1, level=1).copy() 
                    field_data = field_data.ffill()
                    field_data.index.name = 'Dates'
                    
                    file_path = os.path.join(save_dir, field_mappings[field])
                    field_data = self.merge_with_existing_data(field_data, file_path)
                    field_data.to_csv(file_path, sep=";", date_format='%d/%m/%Y')
                    logger.info(f"Saved {field} data to {file_path}")
                except KeyError:
                    logger.warning(f"Field {field} not found in dataframe")
        else:
            if 'Dates' in df.columns:
                df.set_index('Dates', inplace=True)
            
            df = df.ffill()
            df.index.name = 'Dates'
            
            file_path = os.path.join(save_dir, 'price.csv')
            df = self.merge_with_existing_data(df, file_path)
            
            df.to_csv(file_path, sep=';', date_format='%d/%m/%Y')
            logger.info(f"Saved data to {file_path}")
    
    def extract_index_universe_data(self, index_name):
        """Extract data for a given index universe including associated rate data"""
        logger.info(f"Starting extraction for {index_name} universe")
        
        index_upper = index_name.upper()
        index_lower = index_name.lower()
        trading_universe_path = os.path.join(self.base_dir, 'trading_universes', index_lower, 'trading_universe.csv')
        
        try:
            trading_universe = pd.read_csv(trading_universe_path)
            logger.info(f"Loaded trading universe from {trading_universe_path}")
        except FileNotFoundError:
            logger.error(f"Trading universe file not found at: {trading_universe_path}")
            return None
        
        if ' Equity' not in trading_universe['Ticker'].iloc[0]:
            trading_universe['Ticker'] = trading_universe['Ticker'] + ' Equity'
        
        trading_tickers = trading_universe['Ticker'].tolist()
        index_ticker = f"{index_upper} Index"
        all_tickers = [index_ticker] + trading_tickers
        
        logger.info(f"Extracting data for {len(all_tickers)} tickers")
        
        df = self.extract_data_for_universe(all_tickers, list(self.field_mappings.keys()))
        
        if df.empty:
            logger.warning(f"No data extracted for {index_upper} universe")
            return None
        
        index_output_dir = os.path.join(self.output_dir, index_lower)
        self.process_and_save_data(df, all_tickers, index_output_dir, self.field_mappings)
        rate_ticker = self.index_to_rate_mapping.get(index_upper)
        if rate_ticker:
            logger.info(f"Extracting rate data ({rate_ticker}) for {index_upper}")
            rate_df = self.extract_data_for_universe([rate_ticker], ['PX_LAST'])
            
            if not rate_df.empty:
                rate_output_dir = os.path.join(self.output_dir, index_lower)
                self.process_and_save_data(rate_df, [rate_ticker], rate_output_dir, {'PX_LAST': 'rate.csv'})
                logger.info(f"Saved rate data for {index_upper}")
        
        return df
    
    def extract_risk_proxies(self):
        """Extract risk proxy data"""
        logger.info("Starting risk proxies extraction")
        
        unique_tickers = list(dict.fromkeys(self.risk_proxy_tickers))
        logger.info(f"Number of unique risk proxy tickers: {len(unique_tickers)}")
        df = self.extract_data_for_universe(self.risk_proxy_tickers, ['PX_LAST'])
        
        if df.empty:
            logger.warning("No risk proxy data extracted")
            return None
        risk_output_dir = os.path.join(self.output_dir, 'risk_proxies')
        os.makedirs(risk_output_dir, exist_ok=True)
        
        if isinstance(df.columns, pd.MultiIndex):
            combined_df = df.xs('PX_LAST', axis=1, level=1)
        else:
            combined_df = df
        
        combined_df = combined_df.ffill()
        combined_df.index.name = 'Dates'
        combined_file_path = os.path.join(risk_output_dir, 'all_risk_proxies.csv')
        combined_df = self.merge_with_existing_data(combined_df, combined_file_path)
        combined_df.to_csv(combined_file_path, sep=";", date_format='%d/%m/%Y')
        logger.info(f"Saved risk proxies to {combined_file_path}")
        
        return df
    
    def extract_all_indices(self):
        logger.info("Starting all indices extraction")
        df = self.extract_data_for_universe(self.index_tickers, ['PX_LAST'])
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs('PX_LAST', axis=1, level=1)
            
            df = df.ffill()
            df.index.name = 'Dates'
            
            file_path = os.path.join(self.output_dir, 'index_data.csv')
            df = self.merge_with_existing_data(df, file_path)
            df.to_csv(file_path, sep=";", date_format='%d/%m/%Y')
            logger.info(f"Saved index data to {file_path}")
        
        return df


def main():
    parser = argparse.ArgumentParser(description="Bloomberg Data Extraction Tool")
    parser.add_argument('target', choices=['sxxp', 'spx', 'pbh', 'risk_proxies', 'index', 'all'], 
                       help='Target data to extract')
    parser.add_argument('--mode', choices=['full', 'partial'], default='partial',
                       help='Extraction mode (full: 15Y, partial: 100d)')
    parser.add_argument('--base-dir', default=r"X:\Stagiaires\Vincent N\data",
                       help='Base directory for data storage')
    
    args = parser.parse_args()
    
    extractor = BloombergDataExtractor(args.base_dir, args.mode)
    
    logger.info(f"Starting extraction: target={args.target}, mode={args.mode}")
    if args.target == 'sxxp':
        extractor.extract_index_universe_data('SXXP')
    elif args.target == 'spx':
        extractor.extract_index_universe_data('SPX')
    elif args.target == "pbh":
        extractor.extract_index_universe_data("PBH")
    elif args.target == 'risk_proxies':
        extractor.extract_risk_proxies()
    elif args.target == 'index':
        extractor.extract_all_indices()
    elif args.target == 'all':
        extractor.extract_index_universe_data('SXXP')
        extractor.extract_index_universe_data('SPX')
        extractor.extract_risk_proxies()
        extractor.extract_all_indices()
    
    logger.info("Extraction completed!")
    
    print("\nSummary of created/updated files:")
    for root, dirs, files in os.walk(extractor.output_dir):
        for file in files:
            if file.endswith('.csv'):
                filepath = os.path.join(root, file)
                size = os.path.getsize(filepath)
                print(f"{os.path.relpath(filepath, extractor.output_dir)}: {size/1024:.2f} KB")


if __name__ == "__main__":
    main()