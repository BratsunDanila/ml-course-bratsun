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
| **Lasso ($\alpha=0.005$)** 🏆 | **0.9080** | **0.1246** | **$14,350** | **$19,783** | **0.1141** | 1.05 сек |
| **ElasticNet ($\alpha=0.05, l_1=0.1$)** | 0.9064 | 0.1256 | $14,410 | $19,953 | **0.1139** | 1.43 сек |
| **Gradient Boosting (Бонус)** | 0.9038 | 0.1273 | $14,920 | $20,368 | 0.1245 | 2.46 сек |
| **Linear Regression (OLS)** | 0.8968 | 0.1319 | $15,666 | $22,384 | 0.1331 | 0.06 сек |
| **Ridge ($\alpha=300.0$)** | 0.8936 | 0.1339 | $15,310 | $22,009 | 0.1164 | 0.35 сек |

> 🏆 **Лучшая модель:** **Lasso** с параметром регуляризации $\alpha = 0.005$.  
> Модель объясняет **$90.80\%$** дисперсии цен на жильё ($R^2 \approx 0.91$), средняя абсолютная ошибка составляет **~$14,350**, а $L_1$-штраф занулил 187 неинформативных и мультиколлинеарных признаков из 256, сформировав компактное и устойчивое решение.

---

## 🚀 Быстрый старт

```python
import json
import joblib
import requests
from io import BytesIO
import numpy as np
import pandas as pd

# 1. Загрузка обученной модели напрямую из GitHub репозитория
BASE_URL = "https://raw.githubusercontent.com/BratsunDanila/ml-course-bratsun/main/module-2-house-prices"

model = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/best_model.pkl").content))
scaler = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/scaler.pkl").content))
feature_cols = requests.get(f"{BASE_URL}/models/feature_cols.json").json()

# 2. Оценка стоимости дома
# (модель ожидает масштабированный вектор из 256 признаков)
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
│   ├── gbr_model.pkl         # Gradient Boosting Regressor (Бонус)
│   ├── scaler.pkl            # Обученный нормализатор StandardScaler
│   ├── feature_cols.json     # Фиксированный порядок 256 признаков
│   └── metrics.json          # Полный отчёт по метрикам в JSON
└── examples/
    ├── eda_plots.png         # Сетка из 6 графиков EDA и распределения целевой
    ├── predictions_scatter.png# График фактическая vs предсказанная цена
    ├── residuals.png         # Анализ и распределение остатков
    └── feature_importance.png# Топ факторов влияния на стоимость (Lasso)
```
