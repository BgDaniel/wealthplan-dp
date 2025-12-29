import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
b = 0.0005          # baseline hazard
c = 0.09            # aging rate
current_age = 40.0  # starting age (years)

T_years = 60        # horizon
dt = 1.0 / 12.0     # monthly
n_months = int(T_years / dt)

# -----------------------------
# Time and age grids
# -----------------------------
time_grid = np.arange(n_months) * dt          # years since start
age_t = current_age + time_grid               # age at each month

# -----------------------------
# Gompertz hazard (instantaneous)
# -----------------------------
hazard_rates = b * np.exp(c * age_t)

# -----------------------------
# Integrated hazard over each month
# -----------------------------
hazard_integral = (b / c) * (
    np.exp(c * (age_t + dt)) - np.exp(c * age_t)
)

# -----------------------------
# Monthly survival probabilities
# -----------------------------
survival_probs = np.exp(-hazard_integral)

# -----------------------------
# Plot hazard rate
# -----------------------------
plt.figure()
plt.plot(age_t, hazard_rates)
plt.xlabel("Age (years)")
plt.ylabel("Hazard rate λ(age)")
plt.title("Gompertz Hazard Rate")
plt.grid(True)
plt.show()

# -----------------------------
# Plot survival probability
# -----------------------------
plt.figure()
plt.plot(age_t, survival_probs)
plt.xlabel("Age (years)")
plt.ylabel("Monthly survival probability q_t")
plt.title("Monthly Survival Probability (Conditional)")
plt.grid(True)
plt.show()



# -----------------------------
# Total (cumulative) survival probability
# -----------------------------
cumulative_survival = np.cumprod(survival_probs)

# -----------------------------
# Plot cumulative survival
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(age_t, cumulative_survival, lw=2)
plt.xlabel("Age (years)")
plt.ylabel("Cumulative survival probability S(age)")
plt.title("Total Survival Probability under Gompertz Model")
plt.grid(True)
plt.ylim(0,1.05)
plt.show()
