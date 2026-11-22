# Data Sources

All public files must be downloaded **manually** into `data/raw/` (no Kaggle API credentials are needed).
`src/data_prep.py` loads whichever ones are present and skips the rest with a clear message.

| # | Dataset | URL | Licence (check on source page) | Accessed | Status in this repo | How it is used |
|---|---|---|---|---|---|---|
| 1 | Transaction Dataset (ARUNAGIRINATHAN) | https://huggingface.co/datasets/ARUNAGIRINATHAN/Transaction_Dataset | See dataset card (not stated at time of writing) | 2026-11-22 | **Optional, not bundled.** Save as `data/raw/hf_transactions.csv` | External check set for the categorizer after mapping its categories to our taxonomy (`CATEGORY_MAP` in `data_prep.py`) |
| 2 | UPI Transactions 2024 (skullagos5246) | https://www.kaggle.com/datasets/skullagos5246/upi-transactions-2024-dataset | See Kaggle page | 2026-11-22 | **Optional, not bundled.** Save as `data/raw/upi_transactions_2024.csv` | EDA: amount distribution and hour-of-day patterns, compared with our generated amounts |
| 3 | Analyzing Credit Card Spending Habits in India | https://www.kaggle.com/datasets/thedevastator/analyzing-credit-card-spending-habits-in-india | See Kaggle page (CC0 per page at time of writing; verify) | 2026-11-22 | **Optional, not bundled.** Save as `data/raw/credit_card_india.csv` | EDA: expense-type proportions, cross-check for clustering features |
| 4 | HCES 2023-24 Factsheet, MoSPI (official) | https://mospi.gov.in/sites/default/files/publication_reports/HCES%20FactSheet%202023-24.pdf | Government of India open publication | 2026-11-22 | **Used.** Numbers are typed into `HCES_URBAN_SHARES` in `data_prep.py` | Calibrates the generator's category shares (urban) |
| 5 | Budget Allocation (NSSO/MoSPI-derived) | https://baselight.app/u/kaggle/dataset/shrinolo_budget_allocation | See page | 2026-11-22 | Not used (HCES used directly instead) | Secondary cross-check only |
| 6 | Indian Bank Statement Synthetic Dataset (AgamiAI) | https://huggingface.co/datasets/AgamiAI/Indian-Bank-Statements | Apache 2.0 | 2026-11-22 | Not bundled. Narration styles (UPI/NEFT/IMPS/ACH prefixes) were used as a **style reference** for the generator templates | Narration format only, never behaviour |
| 7 | Daily Expense Prediction (IEEE DataPort) | https://ieee-dataport.org/documents/daily-expense-prediction | IEEE DataPort terms | - | Not used | Optional extra examples |
| 8 | Your own bank statement | Your net-banking export | Private | - | **Never commit.** Import with `import_bank_csv()` or the app's CSV upload | Real demo / small real test set |

## HCES 2023-24 (urban) numbers used

Figures from the MoSPI HCES 2023-24 factsheet and press note (share of urban MPCE, %):

| Item group | Share (%) | Source |
|---|---|---|
| Food (total) | 39.68 | Factsheet, Statement 1 |
| Beverages, refreshments & processed food | 11.09 | Press note 2023-24 |
| Conveyance | 8.46 | Press note 2023-24 |
| Misc. goods & entertainment | 6.92 | Press note 2023-24 |
| Durable goods | 6.87 | Press note 2023-24 |
| Rent | 6.58 | Press note 2023-24 |

The remaining groups (medical, education, fuel & light, clothing, toilet articles, consumer services) are **approximations
of the factsheet's Statement 2 values** (each roughly 4-6%). They are marked `approx` in `data_prep.py`;
please verify them against the factsheet PDF and update if needed.

## Honest statement

* No public dataset of real, labelled, multi-month personal transactions for Indian users exists.
* **Every model in this repo is trained on synthetic data** produced by `generate_data()`, calibrated with official HCES shares.
* Reported metrics describe performance on that synthetic data only. They are **not** real-world performance.
* Network access was not available when this repo was built, so datasets #1-#3 were **not** used in the reported runs.
  Drop them into `data/raw/` and re-run `python -m src.data_prep` / the EDA notebook to include them.
