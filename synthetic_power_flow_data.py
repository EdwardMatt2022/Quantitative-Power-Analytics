import pandas as pd
import numpy as np

start_date = '2021-01-04'
end_date = '2024-04-08'
date_range = pd.date_range(start=start_date, end=end_date, freq='H')

power_flow_df = pd.DataFrame(date_range, columns=['Timestamp'])

power_flows = ['Houston_to_North', 'Houston_to_South', 'Houston_to_West', 'South_to_West']

np.random.seed(0)
for flow in power_flows:

    hours = np.arange(len(date_range))
    daily_cycle = 20 * np.sin(2 * np.pi * hours / 24)  # Daily cycle (24-hour period)
    seasonal_cycle = 30 * np.sin(2 * np.pi * hours / (24 * 30))  # Monthly cycle (approx. 30 days)
    

    base_flow = 100  # Base flow in MW
    random_noise = np.random.normal(0, 5, size=len(date_range))  # Random noise with small deviations
    power_flow_df[f'{flow}_Flow_MW'] = base_flow + daily_cycle + seasonal_cycle + random_noise
    

    power_flow_df[f'{flow}_Status'] = np.random.choice([0, 1], size=len(date_range), p=[0.1, 0.9])
    
  
    

file_path = '/Users/edwardmattern/Documents/synthetic_power_flow_data.csv'
power_flow_df.to_csv(file_path, index=False)

print(f"Synthetic power flow data saved as {file_path}")
    