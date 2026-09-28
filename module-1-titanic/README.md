# 🚢 Module 1 — Titanic: EDA + бинарная классификация

**Автор:** Брацун Данила, ПКТб-23-1  
**Учебный год:** 2026/2027  
**Дата:** 2026-09-28  

## 📊 Результаты

| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.80 | 0.78 | 0.68 | 0.73 | 0.85 |
| Decision Tree | 0.78 | 0.74 | 0.67 | 0.70 | 0.81 |
| Random Forest (бонус) | 0.80 | 0.81 | 0.64 | 0.72 | 0.85 |

**Время обучения:** 0.01 сек (LR), 0.00 сек (DT), 0.13 сек (RF)

## 🚀 Быстрый старт

```python
import joblib
import requests
from io import BytesIO

BASE_URL = "https://raw.githubusercontent.com/BratsunDanila/ml-course-bratsun/main/module-1-titanic"
model = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/lr_model.pkl").content))
```

## 📁 Структура
- `notebook.ipynb` — полный отчёт (14 разделов)
- `models/` — модели и метаданные
- `data/` — датасет и описание
- `examples/` — графики и визуализации
- `requirements.txt` — зависимости проекта
