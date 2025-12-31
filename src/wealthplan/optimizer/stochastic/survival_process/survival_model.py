import numpy as np
import matplotlib.pyplot as plt


class SurvivalModel:
    """
    Exponential (Gompertz-style) mortality model.

    Hazard rate:
        h(a) = b * exp(c * a)

    Integrated hazard over [a, a + dt]:
        H(a, a+dt) = (b / c) * (exp(c * (a+dt)) - exp(c * a))
    """

    def __init__(self, b: float, c: float) -> None:
        """
        Initialize mortality model parameters.

        Args:
            b: Level parameter of the hazard rate
            c: Growth rate of the hazard
        """
        self.b: float = b
        self.c: float = c

    def hazard_integral(self, age_t: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute the integrated hazard over each interval of length dt.

        Args:
            age_t: Array of ages
            dt: Time step (e.g., 1/12 for monthly)

        Returns:
            Integrated hazard over [age_t[i], age_t[i]+dt]
        """
        return (self.b / self.c) * (np.exp(self.c * (age_t + dt)) - np.exp(self.c * age_t))

    def conditional_survival_probabilities(self, age_t: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute conditional survival probabilities over one step of length dt.

        q_t = exp(-H(age_t, age_t + dt))

        Args:
            age_t: Array of ages
            dt: Time step (e.g., 1/12 for monthly)

        Returns:
            Array of conditional survival probabilities
        """
        hazard_int = self.hazard_integral(age_t, dt)
        return np.exp(-hazard_int)

    def cumulative_survival_probabilities(self, age_t: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute cumulative survival probabilities starting from age_t[0].

        Args:
            age_t: Array of ages
            dt: Time step (e.g., 1/12 for monthly)

        Returns:
            Array of cumulative survival probabilities
        """
        q_t = self.conditional_survival_probabilities(age_t, dt)
        return np.cumprod(q_t)  # sequential multiplication

    def hazard_rate(self, age_t: np.ndarray) -> np.ndarray:
        """
        Instantaneous hazard rate h(a) = b * exp(c * a)

        Args:
            age_t: Array of ages

        Returns:
            Array of hazard rates
        """
        return self.b * np.exp(self.c * age_t)


if __name__ == "__main__":
    # Gompertz parameters
    b = 9.5e-5
    c = 0.085
    dt_month = 1.0 / 12.0  # monthly step

    model = SurvivalModel(b=b, c=c)

    # Age grid (monthly steps)
    ages = np.arange(30, 100, dt_month)

    # Conditional survival per month
    q_t = model.conditional_survival_probabilities(ages, dt_month)

    # Cumulative survival from start
    S_t = model.cumulative_survival_probabilities(ages, dt_month)

    # Instantaneous hazard
    h_rate = model.hazard_rate(ages)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(ages, q_t, label="Conditional monthly survival q_t")
    plt.plot(ages, S_t, label="Cumulative survival S_t")
    plt.plot(ages, h_rate, label="Instantaneous hazard h(a)")
    plt.xlabel("Age")
    plt.ylabel("Probability / Hazard")
    plt.title("Gompertz Mortality")
    plt.legend()
    plt.grid(True)
    plt.show()
