# -*- coding: utf-8 -*-
"""
Load T_states.csv into pandas DataFrame
"""

import pandas as pd

# Load the T_states.csv file
try:
    df = pd.read_csv('T_states.csv')
    print("Successfully loaded T_states.csv")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    
except FileNotFoundError:
    print("Error: T_states.csv file not found in the current directory")
except Exception as e:
    print(f"Error loading file: {e}")

# Display basic information about the DataFrame
if 'df' in locals():
    print(f"\nDataFrame Info:")
    print(f"Data types: {df.dtypes}")
    print(f"Missing values: {df.isnull().sum().sum()}")
