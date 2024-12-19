#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 20:41:32 2024

@author: edwardmattern
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load import data (replace with actual file paths or APIs)
# Example data: Columns ['Importer', 'Quantity (MMBtu)', 'Price ($/MMBtu)']
data = pd.read_csv("/Users/edwardmattern/Documents/game_theory_model.csv")

# Extract relevant data
firms = data['COMPANY'].unique()
quantities = data.groupby('COMPANY')['NG1_PROD'].sum()
prices = data.groupby('COMPANY')['NG1_PRICE'].mean()

# Parameters for Cournot competition model
# Cost of supplying imports (assume different costs for firms)
costs = {firm: np.random.uniform(1.5, 2.5) for firm in firms}

# Market demand function: P(Q) = a - bQ
a = 12  # Maximum willingness to pay
b = 0.02  # Sensitivity to quantity

# Cournot equilibrium calculation
def calculate_profit(q_i, q_others, cost, firm_price):
    total_quantity = q_i + q_others
    price = firm_price - b * total_quantity  # Adjust price based on firm's base price
    return (price - cost) * q_i

def best_response(q_others, cost, firm_price):
    # Maximize profit with respect to q_i
    best_q = 0
    best_profit = -np.inf
    for q_i in np.linspace(0, 1000, 100):  # Search over possible quantities
        profit = calculate_profit(q_i, q_others, cost, firm_price)
        if profit > best_profit:
            best_profit = profit
            best_q = q_i
    return best_q

# Iteratively solve for Nash equilibrium
nash_quantities = {firm: 0 for firm in firms}
for iteration in range(50):  # Converge over iterations
    for firm in firms:
        others_quantity = sum(nash_quantities[f] for f in firms if f != firm)
        nash_quantities[firm] = best_response(others_quantity, costs[firm], prices[firm])

# Calculate equilibrium price and profits
total_quantity = sum(nash_quantities.values())
equilibrium_prices = {firm: prices[firm] - b * total_quantity for firm in firms}
profits = {firm: calculate_profit(nash_quantities[firm], 
                                  total_quantity - nash_quantities[firm], 
                                  costs[firm], 
                                  prices[firm]) for firm in firms}

# Display results
print("Nash Equilibrium Quantities:")
for firm, quantity in nash_quantities.items():
    print(f"{firm}: {quantity:.2f} MMBtu")

print(f"\nEquilibrium Market Prices:")
for firm, eq_price in equilibrium_prices.items():
    print(f"{firm}: ${eq_price:.2f} per MMBtu")

print("\nFirm Profits:")
for firm, profit in profits.items():
    print(f"{firm}: ${profit:.2f}")

# Visualization
plt.bar(nash_quantities.keys(), nash_quantities.values(), color='brown')
plt.xlabel('Importer')
plt.ylabel('Quantity (MMBtu)')
plt.title('Nash Equilibrium Quantities by Importer')
plt.xticks(rotation=45, ha='right')  # Fix x-axis formatting
plt.tight_layout()  # Ensure labels fit within the figure
plt.show()

# Visualization of Profits
plt.bar(profits.keys(), profits.values(), color='blue')
plt.xlabel('Importer')
plt.ylabel('Profit ($)')
plt.title('Firm Profits at Nash Equilibrium')
plt.xticks(rotation=45, ha='right')  # Fix x-axis formatting
plt.tight_layout()  # Ensure labels fit within the figure
plt.show()