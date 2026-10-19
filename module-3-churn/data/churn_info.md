# Паспорт датасета Telco Customer Churn

## 1. Общие сведения
- **Название:** Telco Customer Churn (IBM Business Analytics Dataset)
- **Источник:** [Kaggle / IBM Sample Data Sets](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Предметная область:** Телекоммуникации, абонентское обслуживание, удержание клиентов (Customer Retention)
- **Тип задачи:** Бинарная классификация с умеренным дисбалансом классов
- **Количество записей:** 7043 клиента
- **Количество колонок:** 21 (1 идентификатор `customerID`, 19 предикторов, 1 целевая переменная `Churn`)

## 2. Структура и распределение целевой переменной
- **Целевая переменная:** `Churn` (факт ухода клиента в течение последнего месяца)
  - `No` (клиент остался): 5174 объекта (73.46%)
  - `Yes` (клиент ушёл): 1869 объектов (26.54%)
- **Соотношение классов:** ~2.77 : 1 (умеренный дисбаланс, требующий стратификации при разбиении, балансировки весов `class_weight='balanced'` и метрик ROC-AUC / PR-AUC / F1-score).

## 3. Спецификация признаков

### Демографические характеристики (4 признака):
1. `gender` (object): пол клиента (`Male`, `Female`).
2. `SeniorCitizen` (int64): признак пенсионера / пожилого возраста (`0` — нет, `1` — да).
3. `Partner` (object): наличие супруга / партнера (`Yes`, `No`).
4. `Dependents` (object): наличие иждивенцев / детей (`Yes`, `No`).

### Подключенные телекоммуникационные сервисы (9 признаков):
5. `PhoneService` (object): подключение к домашней телефонной линии (`Yes`, `No`).
6. `MultipleLines` (object): наличие нескольких телефонных линий (`No phone service`, `No`, `Yes`).
7. `InternetService` (object): тип провайдера интернета (`DSL`, `Fiber optic`, `No`).
8. `OnlineSecurity` (object): услуга антивирусной защиты и сетевой безопасности (`Yes`, `No`, `No internet service`).
9. `OnlineBackup` (object): услуга облачного резервного копирования данных (`Yes`, `No`, `No internet service`).
10. `DeviceProtection` (object): страхование и защита абонентского оборудования (`Yes`, `No`, `No internet service`).
11. `TechSupport` (object): выделенная приоритетная техническая поддержка (`Yes`, `No`, `No internet service`).
12. `StreamingTV` (object): пакет стримингового цифрового телевидения (`Yes`, `No`, `No internet service`).
13. `StreamingMovies` (object): пакет онлайн-кинотеатра и стриминга фильмов (`Yes`, `No`, `No internet service`).

### Информация об абонентском договоре и оплате (6 признаков):
14. `tenure` (int64): продолжительность обслуживания клиента в месяцах (от 0 до 72 месяцев).
15. `Contract` (object): периодичность контрактных обязательств (`Month-to-month`, `One year`, `Two year`).
16. `PaperlessBilling` (object): электронный безналичный биллинг / выставление счетов (`Yes`, `No`).
17. `PaymentMethod` (object): способ оплаты (`Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)`).
18. `MonthlyCharges` (float64): текущий размер ежемесячного платежа в долларах США ($).
19. `TotalCharges` (object / float64): суммарный объем платежей за весь период обслуживания в долларах США ($). Содержит 11 пробельных значений ` ` для новых клиентов с `tenure = 0`, подлежащих преобразованию в `np.nan` и заполнению медианой.

### Системный идентификатор (1 признак):
20. `customerID` (object): уникальный строковый номер контракта абонента (исключается из обучения).

## 4. Конструируемые признаки (Feature Engineering)
- `TenureYears`: стаж абонента в годах (`tenure / 12`).
- `AvgMonthlyCharge`: расчетный средний чек за месяц (`TotalCharges / (tenure + 1)`).
- `IsNewClient`: бинарный флаг клиента с высоким риском раннего оттока (`tenure <= 6`).
- `HasMultipleServices`: суммарное количество активных цифровых услуг (от 0 до 9).
