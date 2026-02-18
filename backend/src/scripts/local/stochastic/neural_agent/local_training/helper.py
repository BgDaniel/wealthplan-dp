import math

delta_t = 1.0 / 12.0

yearly_savings_return = 0.04

monthly_savings_return = (1.0 + yearly_savings_return) ** (1.0 / 12.0) - 1.0
continues_monthly_savings_return = math.log(1.0 + monthly_savings_return) / delta_t

sigma = 0.05





monthly_stock_return = (continues_monthly_savings_return + sigma ** 2) * delta_t


yearly_stock_return = (1.0 + monthly_stock_return) ** 12 - 1.0

print(yearly_stock_return)

