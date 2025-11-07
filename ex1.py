import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# Дані часового ряду
data = [1.6, 0.8, 1.2, 0.5, 0.9, 1.1, 1.1, 0.6, 1.5, 0.8, 0.9, 1.2, 0.5, 1.3, 0.8, 1.2]
n = len(data)

print(f"Часовий ряд: {data}")
print(f"Кількість спостережень: {n}")

# а) Побудова графіка часового ряду
plt.figure(figsize=(15, 10))

# Графік 1: Основний часовий ряд
plt.subplot(2, 2, 1)
plt.plot(range(1, n+1), data, 'bo-', linewidth=2, markersize=6, label='Часовий ряд')
plt.xlabel('Час (t)')
plt.ylabel('Значення y(t)')
plt.title('а) Графік часового ряду')
plt.grid(True, alpha=0.3)
plt.legend()

# б) Графік для наближеного визначення автокореляції
plt.subplot(2, 2, 2)
plt.plot(range(1, n+1), data, 'bo-', linewidth=2, markersize=6)
plt.xlabel('Час (t)')
plt.ylabel('Значення y(t)')
plt.title('б) Визначення автокореляції (візуально)')
plt.grid(True, alpha=0.3)

# Додаємо лінії для візуалізації зв'язку між сусідніми значеннями
for i in range(1, n):
    plt.plot([i, i+1], [data[i-1], data[i]], 'r--', alpha=0.5)

# в) Графік залежності y(t+1) від y(t)
plt.subplot(2, 2, 3)
y_t = data[:-1]  # y(t)
y_t1 = data[1:]  # y(t+1)

plt.scatter(y_t, y_t1, color='red', s=50, alpha=0.7)
plt.xlabel('y(t)')
plt.ylabel('y(t+1)')
plt.title('в) Залежність y(t+1) від y(t)')
plt.grid(True, alpha=0.3)

# Додаємо лінію тренду
z = np.polyfit(y_t, y_t1, 1)
p = np.poly1d(z)
plt.plot(y_t, p(y_t), "b--", alpha=0.8, label=f'Лінія тренду: y = {z[0]:.3f}x + {z[1]:.3f}')
plt.legend()

# г) Точний розрахунок автокореляції
plt.subplot(2, 2, 4)

# Розрахунок автокореляції
def calculate_autocorrelation(data, lag=1):
    n = len(data)
    mean = np.mean(data)
    
    # Коваріація
    covariance = sum((data[i] - mean) * (data[i+lag] - mean) for i in range(n - lag))
    
    # Дисперсія
    variance = sum((x - mean) ** 2 for x in data)
    
    return covariance / variance

# Автокореляція першого порядку
autocorr_approx = z[0]  # Наближено з графіка
autocorr_exact = calculate_autocorrelation(data, lag=1)

# Використання pandas для перевірки
series = pd.Series(data)
autocorr_pandas = series.autocorr(lag=1)

plt.bar(['Наближено', 'Точний розрахунок', 'Pandas'], 
        [autocorr_approx, autocorr_exact, autocorr_pandas], 
        color=['orange', 'green', 'blue'], alpha=0.7)
plt.ylabel('Коефіцієнт автокореляції')
plt.title('г) Порівняння методів розрахунку')
plt.grid(True, alpha=0.3)

# Додаємо значення на стовпці
for i, v in enumerate([autocorr_approx, autocorr_exact, autocorr_pandas]):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Вивід результатів
print("\nРЕЗУЛЬТАТИ АНАЛІЗУ:")
print(f"Наближений коефіцієнт автокореляції (з графіка): {autocorr_approx:.4f}")
print(f"Точний коефіцієнт автокореляції: {autocorr_exact:.4f}")
print(f"Коефіцієнт автокореляції (pandas): {autocorr_pandas:.4f}")
print(f"Середнє значення ряду: {np.mean(data):.4f}")
print(f"Стандартне відхилення: {np.std(data):.4f}")