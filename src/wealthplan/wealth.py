class Wealth:
    """Represents the wealth with an initial value and yearly return."""

    def __init__(self, initial_wealth: float, yearly_return: float) -> None:
        """
        Args:
            initial_wealth (float): Initial wealth amount.
            yearly_return (float): Annual return as a decimal (e.g., 0.05 for 5%).
        """
        self.initial_wealth = initial_wealth
        self.yearly_return = yearly_return

    def monthly_return(self) -> float:
        """Convert yearly return to a monthly equivalent."""
        return (1 + self.yearly_return) ** (1/12) - 1
