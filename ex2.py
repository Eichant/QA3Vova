import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class RandomWalkWithDrift:
    """Модель випадкового блукання з напрямом"""
    
    def __init__(self, drift=0.1, volatility=0.5, initial_value=0):
        self.drift = drift  # Сталий тренд (напрям)
        self.volatility = volatility  # Волатильність
        self.initial_value = initial_value
    
    def generate_series(self, n_periods=100):
        """Генерація часового ряду"""
        values = [self.initial_value]
        for t in range(1, n_periods):
            # Y(t) = Y(t-1) + drift + ε(t), де ε(t) ~ N(0, σ²)
            shock = np.random.normal(0, self.volatility)
            new_value = values[-1] + self.drift + shock
            values.append(new_value)
        return values
    
    def forecast(self, current_value, horizon):
        """Прогноз на horizon періодів вперед"""
        # Прогноз: Ŷ(t+τ) = Y(t) + τ * drift
        forecast_value = current_value + horizon * self.drift
        return forecast_value
    
    def forecast_error(self, horizon):
        """Прогнозна помилка та середня квадратична похибка"""
        # Помилка прогнозу: e(t+τ) = Σ(ε(t+i)) для i=1 до τ
        # MSE = τ * σ²
        forecast_error_variance = horizon * (self.volatility ** 2)
        mse = forecast_error_variance
        return forecast_error_variance, mse

# Демонстрація моделі
np.random.seed(42)  # Для відтворюваності
model = RandomWalkWithDrift(drift=0.1, volatility=0.5, initial_value=10)

# Генерація даних
n_periods = 50
time_series = model.generate_series(n_periods)

# Прогнозування
current_value = time_series[-1]
forecast_horizon = 10
forecast_values = [model.forecast(current_value, tau) for tau in range(forecast_horizon + 1)]

# Розрахунок помилок
forecast_errors = []
mses = []
for tau in range(1, forecast_horizon + 1):
    error_var, mse = model.forecast_error(tau)
    forecast_errors.append(error_var)
    mses.append(mse)

# Візуалізація
plt.figure(figsize=(15, 10))

# Графік 1: Часовий ряд та прогноз
plt.subplot(2, 2, 1)
time_index = list(range(n_periods))
forecast_index = list(range(n_periods, n_periods + forecast_horizon + 1))

plt.plot(time_index, time_series, 'b-', linewidth=2, label='Спостережуваний ряд')
plt.plot(forecast_index, forecast_values, 'r--', linewidth=2, label='Прогноз')
plt.axvline(x=n_periods, color='gray', linestyle=':', alpha=0.7)
plt.xlabel('Час (t)')
plt.ylabel('Y(t)')
plt.title('Випадкове блукання з напрямом та прогноз')
plt.legend()
plt.grid(True, alpha=0.3)

# Графік 2: Демонстрація тренду
plt.subplot(2, 2, 2)
# Генеруємо кілька траєкторій для демонстрації
n_simulations = 5
for i in range(n_simulations):
    sim_series = model.generate_series(n_periods)
    plt.plot(time_index, sim_series, alpha=0.6)
plt.xlabel('Час (t)')
plt.ylabel('Y(t)')
plt.title('Кілька реалізацій процесу')
plt.grid(True, alpha=0.3)

# Графік 3: Прогнозна помилка
plt.subplot(2, 2, 3)
tau_values = list(range(1, forecast_horizon + 1))
plt.plot(tau_values, forecast_errors, 'ro-', linewidth=2)
plt.xlabel('Горизонт прогнозу (τ)')
plt.ylabel('Дисперсія помилки прогнозу')
plt.title('Залежність помилки прогнозу від горизонту')
plt.grid(True, alpha=0.3)

# Графік 4: Середня квадратична похибка
plt.subplot(2, 2, 4)
plt.plot(tau_values, mses, 'go-', linewidth=2)
plt.xlabel('Горизонт прогнозу (τ)')
plt.ylabel('MSE')
plt.title('Середня квадратична похибка прогнозу')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Теоретичні викладки
print("\nТЕОРЕТИЧНІ ВИКЛАДКИ:")
print("Модель: Y(t) = Y(t-1) + μ + ε(t), де ε(t) ~ N(0, σ²)")
print(f"Параметри моделі: μ = {model.drift}, σ = {model.volatility}")
print(f"Початкове значення: Y(0) = {model.initial_value}")

print("\nПРОГНОЗ:")
print(f"Останнє спостережуване значення: Y({n_periods}) = {current_value:.4f}")
for tau in [1, 5, 10]:
    forecast_val = model.forecast(current_value, tau)
    error_var, mse = model.forecast_error(tau)
    print(f"Прогноз на {tau} періодів: Ŷ({n_periods}+{tau}) = {forecast_val:.4f}")
    print(f"  Дисперсія помилки: {error_var:.4f}")
    print(f"  MSE: {mse:.4f}")
