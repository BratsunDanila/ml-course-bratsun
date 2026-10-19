# Module 3 — Customer Churn: Ансамблевые методы

**Автор:** Брацун Данила, ПКТб-23-1  
**Преподаватель:** Спирин И.С.  
**Учебный год:** 2026/2027  
**Дата сдачи:** 19.10.2026  

## Цель работы
Освоение полного цикла ML-проекта бинарной классификации оттока клиентов на основе ансамблевых методов: глубокий разведочный анализ данных (EDA), обработка пропусков и Feature Engineering (`TenureYears`, `AvgMonthlyCharge`, `IsNewClient`, `HasMultipleServices`), обучение и сопоставление 4 ансамблевых моделей (`Random Forest`, `XGBoost`, `LightGBM`, `CatBoost`), байесовская оптимизация гиперпараметров через `Optuna`, интерпретация решений через значения Шепли (`SHAP`), реализация бонусных техник (`Stacking Classifier`, `SMOTE`, калибровка вероятностей `CalibratedClassifierCV`) и развертывание инференс-пайплайна.

---

## Результаты

Оценка моделей проводилась на тестовой выборке ($20\%$, $n=1409$, `stratify=y`, `random_state=42`):

| Модель | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Время обучения |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Stacking (Бонус)** | **0.7630** | **0.5392** | 0.7487 | **0.6269** | **0.8465** | ~3.50 сек |
| **LightGBM (Optuna)** | 0.7544 | 0.5253 | **0.7781** | **0.6272** | **0.8454** | ~0.18 сек |
| **CatBoost** | 0.7488 | 0.5187 | 0.7433 | 0.6110 | 0.8409 | ~1.00 сек |
| **Random Forest** | 0.7601 | 0.5346 | 0.7433 | 0.6219 | 0.8388 | ~0.40 сек |
| **LightGBM Baseline** | 0.7594 | 0.5332 | 0.7513 | 0.6238 | 0.8377 | **~0.15 сек** |
| **LightGBM + SMOTE (Бонус)** | 0.7828 | 0.5841 | 0.6417 | 0.6115 | 0.8374 | ~0.25 сек |
| **XGBoost** | 0.7622 | 0.5397 | 0.7086 | 0.6127 | 0.8262 | ~0.31 сек |

> **Лучшая одиночная модель:** **LightGBM (Optuna)** — максимальный баланс качества ранжирования ($ROC\text{-}AUC = 0.8454$), высокого охвата уходящих абонентов ($Recall = 77.81\%$) и рекордной скорости инференса.  
> **Лучший мета-ансамбль:** **Stacking Classifier** ($ROC\text{-}AUC = 0.8465$) на базе разнородных базовых моделей.

---

## Быстрый старт

```python
import joblib
import requests
from io import BytesIO

BASE_URL = "https://raw.githubusercontent.com/BratsunDanila/ml-course-bratsun/main/module-3-churn"

# 1. Загрузка обученной модели и списка признаков
model = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/best_model.pkl").content))
feature_cols = requests.get(f"{BASE_URL}/models/feature_cols.json").json()

# 2. Прогнозирование вероятности оттока
# y_proba = model.predict_proba(sample_data[feature_cols])[:, 1]
```

---

## Структура модуля

```
module-3-churn/
├── README.md                 # Описание модуля и сводная таблица метрик
├── notebook.ipynb            # Академический отчёт (14 разделов + Бонус, 0 ошибок)
├── requirements.txt          # Фиксированные зависимости проекта
├── data/
│   ├── telco_churn.csv       # Обучающая выборка (7043 строки, 21 колонка)
│   └── churn_info.md         # Паспорт и спецификация признаков датасета
├── models/
│   ├── best_model.pkl        # Лучшая модель (LightGBM Optuna)
│   ├── rf_model.pkl          # Обученная модель Random Forest
│   ├── xgb_model.pkl         # Обученная модель XGBoost
│   ├── lgbm_model.pkl        # Обученная модель LightGBM Baseline
│   ├── cat_model.pkl         # Обученная модель CatBoost
│   ├── stacking_model.pkl    # Обученный Stacking Classifier (Бонус)
│   ├── scaler.pkl            # Обученный нормализатор StandardScaler
│   ├── label_encoders.pkl    # Словарь LabelEncoder для категориальных шкал
│   ├── feature_cols.json     # Фиксированный порядок 23 признаков
│   └── metrics.json          # Полный структурированный отчёт по метрикам в JSON
└── examples/
    ├── eda_plots.png         # Сетка из 6 графиков EDA
    ├── confusion_matrix.png  # Матрица ошибок лучшей модели
    ├── roc_curves.png        # Сводный график ROC-кривых всех моделей
    ├── feature_importance.png# Топ-15 факторов оттока (LightGBM)
    ├── shap_summary.png      # SHAP Beeswarm Summary Plot
    ├── shap_waterfall.png    # SHAP Waterfall разбор абонента с высоким риском
    └── shap_dependence.png   # SHAP Dependence Plot по стажу tenure
```
