# Data Sources

Public files live in `data/raw/` under the names below. `src/data_prep.py` loads whichever are present and
skips the rest with a clear message. No Kaggle API credentials are needed.

| # | Dataset | URL | Licence | Status | File in `data/raw/` | How it is used |
|---|---|---|---|---|---|---|
| 1 | Transaction Dataset (ARUNAGIRINATHAN) | https://huggingface.co/datasets/ARUNAGIRINATHAN/Transaction_Dataset | Not stated on the card when we checked: **verify** | **Used** | `hf_transactions.csv` (1,000 rows) | (a) 29 real merchant names added to the generator; (b) 422 debit rows added to categorizer training; (c) 423 debit rows held out as an **external test set** |
| 2 | UPI Transactions 2024 (skullagos5246) | https://www.kaggle.com/datasets/skullagos5246/upi-transactions-2024-dataset | See Kaggle page: **verify** | **Used** | `upi_transactions_2024.csv` (250,000 rows) | Calibrates the generator: ticket-size distribution (median, log-sd) and weekend pattern per category. Saved to `data/processed/upi_calibration.csv`. EDA comparison |
| 3 | Analyzing Credit Card Spending Habits in India | https://www.kaggle.com/datasets/thedevastator/analyzing-credit-card-spending-habits-in-india | See Kaggle page: **verify** | **Used (EDA)** | `credit_card_india.csv` (26,052 rows) | EDA cross-check of category spend shares (vs generated and HCES) and spend-mix differences by city / card type (supports clustering) |
| 4 | HCES 2023-24 Factsheet, MoSPI (official) | https://mospi.gov.in/sites/default/files/publication_reports/HCES%20FactSheet%202023-24.pdf | Government of India open publication | **Used** | numbers typed into `HCES_URBAN_SHARES` in `data_prep.py` | Generator category shares (urban) |
| 5 | Budget Allocation (NSSO/MoSPI-derived) | https://baselight.app/u/kaggle/dataset/shrinolo_budget_allocation | See page | Not used | - | HCES used directly instead |
| 6 | Indian Bank Statement Synthetic Dataset (AgamiAI) | https://huggingface.co/datasets/AgamiAI/Indian-Bank-Statements | Apache 2.0 | Not downloaded | - | Narration formats (UPI/NEFT/IMPS/ACH prefixes) written from general knowledge of this style |
| 7 | Daily Expense Prediction (IEEE DataPort) | https://ieee-dataport.org/documents/daily-expense-prediction | IEEE DataPort terms | Not used | - | Optional |
| 8 | Your own bank statement | Your net-banking export | Private | - | **never commit** | App CSV upload / `import_bank_csv()` |

> The public CSVs are committed so a clean clone reproduces the reported results. If a dataset's licence does not allow
> redistribution, delete the file, add it back to `.gitignore`, and ask users to download it from the URL above.

## Real columns found (inspected 2026-11-22)

**#1 HF Transaction Dataset**: `date, description, amount, category`
* `description` = merchant + id, e.g. `Dentist txn_200` (the `txn` id is removed during cleaning)
* 7 categories: Income 155, Food 153, Bills 150, Shopping 139, Travel 138, Entertainment 134, Health 131. 52 merchants, dates 2025-01..2026-01
* All amounts are positive, and `Income` rows are treated as credits. No payment-mode prefixes.
* **Mapping** (`HF_MERCHANT_MAP`, then `HF_CATEGORY_MAP`): its coarse labels disagree with our taxonomy for some merchants, so
  mapping is per merchant first: Blinkit / BigBasket / Local Grocery → Groceries; Netflix / Prime Video / Disney+ Hotstar /
  Spotify → Subscriptions; Gym Membership → Personal Care; Ola / Uber / Auto Rickshaw / Parking / Shell Fuel / Petrol Pump →
  Transport; IndiGo / IRCTC → Travel; Monthly Salary → Salary. The rest follow their category (Food → Food & Dining,
  Bills → Bills & Utilities, Income → Other Income, ...).

**#2 UPI Transactions 2024**: `transaction id, timestamp, transaction type, merchant_category, amount (INR),
transaction_status, sender_age_group, receiver_age_group, sender_state, sender_bank, receiver_bank, device_type,
network_type, fraud_flag, hour_of_day, day_of_week, is_weekend`
* 10 merchant categories mapped by `UPI_CATEGORY_MAP` (Grocery → Groceries, Fuel + Transport → Transport, Utilities →
  Bills & Utilities, Healthcare → Health, Other → Others, ...). Only `SUCCESS` rows are used (237,624 of 250,000).
* Median tickets: Transport ₹539, Entertainment ₹294, Food ₹332, Health ₹386, Others ₹615, Groceries ₹819,
  Shopping ₹1,566, Utilities ₹1,861, Education ₹3,667.
* **Weekend-to-weekday ratio is 0.98-1.05 in every category, so this (simulated) dataset has no weekend effect.** The
  generator therefore no longer assumes higher weekend spending.

**#3 Credit card India**: `index, City, Date, Card Type, Exp Type, Gender, Amount`
* 6 expense types (Food, Fuel, Bills, Entertainment, Grocery, Travel), 4 card types, 986 cities, 2013-2015.
  Aggregated rows, not per-user histories, so it is used for EDA only.

## HCES 2023-24 (urban) numbers used

| Item group | Share of MPCE (%) | Source |
|---|---|---|
| Food (total) | 39.68 | Factsheet, Statement 1 |
| Beverages, refreshments & processed food | 11.09 | Press note 2023-24 |
| Conveyance | 8.46 | Press note 2023-24 |
| Misc. goods & entertainment | 6.92 | Press note 2023-24 |
| Durable goods | 6.87 | Press note 2023-24 |
| Rent | 6.58 | Press note 2023-24 |

Other groups (medical, education, fuel & light, clothing, toilet articles, consumer services) are **approximations**
(each roughly 4-6%), marked `approx` in `data_prep.py`. Verify them against the factsheet PDF.

## Honest statement

* No public dataset of real, labelled, multi-month personal transactions for Indian users exists.
* Public datasets #1-#3 are themselves **simulated or synthetic**. They supply realistic merchant names, ticket sizes and
  a labelled external test set, but not real people's behaviour.
* Forecasting, subscription detection and clustering need multi-month per-user histories, which only our generator provides.
* Reported metrics describe performance on this data only, not real-world performance.
