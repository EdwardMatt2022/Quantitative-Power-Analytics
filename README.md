The goal of this project is to model energy demand (MW) in ERCOT with features that are correlated with load demand.
Explanation of the features: Total of 11 commodity prices, these are categorized into three groups, fuel prices, zonal electricity prices, and the carbon allowance price.
There are also features for natural gas storage and liquefied natural gas for natural gas trading models.
Lastly, I have features for weather variables, maximum temperature and average wind speed to list a few.
The performance metric used in this project would be MAPE and MAE, the reason why MSE is not used is because it performs poorly against seasonal data such as electricity demand.


