# Module 4 — Online Shoppers: Инженерия признаков и работа с нестандартными данными

**Автор:** Брацун Данила, ПКТб-23-1  
**Преподаватель:** Спирин И.С.  
**Учебный год:** 2026/2027  
**Дата сдачи:** 02.11.2026  

## Цель работы
Освоение полного прикладного цикла инженерии признаков (Feature Engineering) для структурированных данных с элементами веб-аналитики и поведения пользователей интернет-магазина (Online Shoppers Purchasing Intention Dataset, UCI ID: 468). Работа включает: всесторонний разведочный анализ данных (EDA), конструирование доменных поведенческих метрик (`TotalPages`, `TotalDuration`, `AvgPageValue`, `BounceExitRatio`, `ProductPageRatio`, `AvgDurationPerPage`, `PageValues_x_TotalDuration`), обучение и сопоставление 4 ключевых ансамблевых алгоритмов (`Random Forest`, `XGBoost`, `LightGBM`, `CatBoost`) с компенсацией выраженного дисбаланса классов (84.5% / 15.5%), байесовскую оптимизацию гиперпараметров через `Optuna`, глубинную интерпретацию факторов конверсии через значения Шепли (`SHAP`), реализацию бонусных техник (+5 баллов: мета-ансамбль `Stacking Classifier`, синтетическую балансировку `SMOTE`, калибровку вероятностей `CalibratedClassifierCV` с минимизацией Brier Score) и автономный инференс-пайплайн.

---

## Результаты

Оценка качества моделей проводилась на отложенной тестовой выборке ($20\%$, $n=2466$, стратифицированное разбиение `stratify=y`, `random_state=42`):

| Модель | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Время обучения |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LightGBM (Optuna)** | **0.8715** | 0.5573 | **0.8272** | **0.6660** | **0.9335** | ~0.42 сек |
| **Stacking (Бонус)** | 0.8642 | 0.5396 | **0.8377** | 0.6564 | **0.9330** | ~16.69 сек |
| **CatBoost** | 0.8678 | 0.5513 | 0.7880 | 0.6487 | 0.9276 | ~1.52 сек |
| **LightGBM Baseline** | 0.8715 | 0.5603 | 0.7906 | 0.6558 | 0.9276 | **~0.31 сек** |
| **LightGBM + SMOTE (Бонус)** | **0.8921** | **0.6324** | 0.7251 | **0.6756** | 0.9259 | ~0.36 сек |
| **XGBoost** | 0.8747 | 0.5712 | 0.7670 | 0.6547 | 0.9255 | ~0.48 сек |
| **Random Forest** | 0.8727 | 0.5632 | 0.7932 | 0.6587 | 0.9249 | ~0.61 сек |

> **Лучшая одиночная модель:** **LightGBM (Optuna)** — максимальное качество ранжирования сессий ($ROC\text{-}AUC = 0.9335$, CV $= 0.9355$), высокий охват покупателей ($Recall = 82.72\%$) и рекордная скорость обучения.  
> **Лучший мета-ансамбль:** **Stacking Classifier** ($ROC\text{-}AUC = 0.9330$, $Recall = 83.77\%$).  
> **Калибровка вероятностей (Бонус):** Применение изотонической калибровки снизило квадратичную ошибку вероятностей Brier Score с **0.0976** до **0.0693** (повышение надёжности оценки конверсии на 29%).

---

## Быстрый старт

```python
import joblib
import requests
from io import BytesIO

BASE_URL = "https://raw.githubusercontent.com/BratsunDanila/ml-course-bratsun/main/module-4-features"

# 1. Загрузка обученной модели и списка признаков
model = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/best_model.pkl").content))
feature_cols = requests.get(f"{BASE_URL}/models/feature_cols.json").json()

# 2. Прогнозирование вероятности совершения покупки
# y_proba = model.predict_proba(sample_sessions[feature_cols])[:, 1]
```

---

## Структура модуля

```
module-4-features/
├── README.md                 # Описание модуля и сводная таблица метрик
├── notebook.ipynb            # Академический отчёт (14 разделов + Бонусы, 0 ошибок)
├── requirements.txt          # Фиксированные зависимости проекта
├── data/
│   ├── online_shoppers_intention.csv  # Выборка сессий (12 330 строк, 18 колонок)
│   └── dataset_info.md                # Паспорт и спецификация признаков
├── models/
│   ├── best_model.pkl        # Лучшая оптимизированная модель (LightGBM Optuna)
│   ├── rf_model.pkl          # Обученная модель Random Forest
│   ├── xgb_model.pkl         # Обученная модель XGBoost
│   ├── lgbm_model.pkl        # Обученная модель LightGBM Baseline
│   ├── cat_model.pkl         # Обученная модель CatBoost
│   ├── stacking_model.pkl    # Обученный Stacking Classifier (Бонус)
│   ├── scaler.pkl            # Обученный нормализатор StandardScaler
│   ├── feature_cols.json     # Фиксированный порядок 27 признаков
│   └── metrics.json          # Полный структурированный отчёт по метрикам в JSON
└── examples/
    ├── eda_plots.png         # Сетка из 6 графиков разведочного анализа данных
    ├── confusion_matrix.png  # Матрица ошибок лучшей модели
    ├── roc_curves.png        # Сводный график ROC-кривых всех моделей
    ├── feature_importance.png# Топ факторов влияния (LightGBM vs Random Forest)
    ├── shap_summary.png      # SHAP Beeswarm Summary Plot
    ├── shap_waterfall.png    # SHAP Waterfall разбор сессии с высокой вероятностью
    └── shap_dependence.png   # SHAP Dependence Plot по ключевому признаку PageValues
```
