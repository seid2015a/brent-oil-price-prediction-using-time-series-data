import pandas as pd
import numpy as np

def load_brent_data(filepath):
    #load brent oil price data from a CSV data
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date']) 
        df = df.sort_values(by = 'Date').reset_index(drop= True) 
        return df
    except FileNotFoundError:
        print(f'Error: file not found at {filepath}')
        return None
    except KeyError:
        print("Error: 'Date' or 'price' column do not found in the CSV")
        return None
    except Exception as e:
        print(f"An error occured during data loading: {e}")
        return None
    
def calculate_log_returns(df, price_col='Price'):
    """
    Calculates log returns from a price series.
    log_return_t = log(price_t / price_{t-1})
    """
    if price_col not in df.columns:
        print(f"Error: price column '{price_col} not found in DataFrame")
        return None
    df['Log_Return']= np.log(df[price_col]/df[price_col].shift(1))
    return df.dropna().reset_index(drop = True)   #dromp the first NaN row
def load_event_data(filepath):
    """
    Loads key event data from a CSV file.
    Converts 'Date' column to datetime.
    """
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by = 'Date').reset_index(drop=True)
        return df
    except FileNotFoundError:
        print(f"Error: file not found at {filepath}.")
        return None
    except KeyError:
        print("Error: 'Date', 'EventType' or 'EventDescription' not found in the CSV.")
        return None
    except Exception as e:
        print(f"An error occuring during data loading: {e}")
        return None
    
if __name__ == '__main__':
    brent_df = load_brent_data('../data/raw/BrentOilPrices.csv')
    
    if brent_df is not None:
        print("Brent Data Head:")
        print(brent_df.head())
        print('\nBrent Data Info:')
        brent_df.info()
    
        brent_df_with_returns = calculate_log_returns(brent_df.copy())
        if brent_df_with_returns is not None:
            print("Brent Data with Log Reterns Head: ")
            print(brent_df_with_returns.head())
            print("\n Log Return Statics: ")
            print(brent_df_with_returns['Log_Return'].describe())
            
    event_df = load_event_data('../data/raw/key_events.csv')
    if event_df is not None:
        print("Event Data Head:")
        print(event_df.head())
        print("\nEvent Data Info:")
        event_df.info()
        