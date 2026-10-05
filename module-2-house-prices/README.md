# 🏠 Module 2 — House Prices: Предобработка данных и регрессия

**Автор:** Брацун Данила, ПКТб-23-1  
**Учебный год:** 2026/2027  
**Дата:** 2026-10-05  

## 🎯 Цель работы
Освоение полного цикла ML-проекта для задачи регрессии: глубокая очистка данных и восстановление пропусков, логарифмирование скошенной целевой переменной, конструирование агрегированных признаков (`TotalSF`, `TotalBath`, `HouseAge`), обучение моделей с $L_1$- и $L_2$-регуляризацией (`Ridge`, `Lasso`, `ElasticNet`), кросс-валидация и подбор гиперпараметров через `GridSearchCV`.

---

## 📊 Результаты

Оценка моделей проводилась на тестовой выборке ($20\%$, $n=292$, `random_state=42`) и с помощью 5-Fold кросс-валидации:

| Модель | $R^2$ (Test) | RMSE (log) | MAE ($) | RMSE ($) | CV RMSE (log) | Время обучения |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Lasso ($\alpha=0.005$)** 🏆 | **0.9099** | **0.1233** | **$14,313** | **$19,839** | **0.1141** | 1.15 сек |
| **ElasticNet ($\alpha=0.05, l_1=0.1$)** | 0.9071 | 0.1251 | $14,374 | $20,105 | **0.1139** | 3.83 сек |
| **Linear Regression (OLS)** | 0.8969 | 0.1318 | $15,755 | $22,475 | 0.1331 | 0.17 сек |
| **Ridge ($\alpha=300.0$)** | 0.8923 | 0.1348 | $15,555 | $22,258 | 0.1164 | 0.85 сек |

> 🏆 **Лучшая модель:** **Lasso** с параметром регуляризации $\alpha = 0.005$.  
> Модель объясняет **$90.99\%$** дисперсии цен на жильё ($R^2 = 0.91$), средняя абсолютная ошибка составляет **$14,313**, а $L_1$-штраф занулил 186 неинформативных и мультиколлинеарных признаков из 252, сформировав компактное и устойчивое решение.

---

## 🚀 Быстрый старт

```python
import json
import joblib
import numpy as np
import pandas as pd

# 1. Загрузка обученной модели, нормализатора и списка признаков
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
with open('models/feature_cols.json', 'r', encoding='utf-8') as f:
    feature_cols = json.load(f)

# 2. Пример: предсказание стоимости дома
sample_house = pd.DataFrame([{col: 0 for col in feature_cols}])
sample_house['TotalSF'] = 2500
sample_house['OverallQual'] = 8
sample_house['HouseAge'] = 5
sample_house['TotalBath'] = 3.0

# 3. Инференс
sample_scaled = scaler.transform(sample_house[feature_cols])
pred_log = model.predict(sample_scaled)[0]
pred_price = np.expm1(pred_log)

print(f"Оценочная стоимость дома: ${pred_price:,.2f}")
```

---

## 📁 Структура модуля

```
module-2-house-prices/
├── README.md                 # Описание модуля и сводная таблица метрик
├── notebook.ipynb            # Академический отчёт (14 разделов, 0 ошибок/предупреждений)
├── requirements.txt          # Фиксированные зависимости проекта
├── submission.csv            # Файл предсказаний для Kaggle (1459 строк)
├── data/
│   ├── train.csv             # Обучающая выборка (1460 строк)
│   ├── test.csv              # Тестовая выборка (1459 строк)
│   └── house_prices_info.md  # Паспорт и спецификация признаков
├── models/
│   ├── best_model.pkl        # Лучшая модель (Lasso)
│   ├── lr_model.pkl          # Linear Regression
│   ├── ridge_model.pkl       # Обученная модель Ridge
│   ├── lasso_model.pkl       # Обученная модель Lasso
│   ├── enet_model.pkl        # Обученная модель ElasticNet
│   ├── scaler.pkl            # Обученный нормализатор StandardScaler
│   ├── feature_cols.json     # Фиксированный порядок 252 признаков
│   └── metrics.json          # Полный отчёт по метрикам в JSON
└── examples/
    ├── eda_plots.png         # Анализ распределения целевой и выбросов
    ├── predictions_scatter.png# График фактическая vs предсказанная цена
    ├── residuals.png         # Анализ и распределение остатков
    └── feature_importance.png# Топ факторов влияния на стоимость
```
