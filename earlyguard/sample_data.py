"""
Creates small sample CSVs (50 accounts) for a fast smoke test.
"""
import pandas as pd, os

DATA = "data"
OUT  = "data/sample"
os.makedirs(OUT, exist_ok=True)

# Load loan accounts and pick first 50 account IDs
loan = pd.read_csv(f"{DATA}/loan_accounts.csv")
ids  = loan['account_id'].head(50).tolist()
loan.head(50).to_csv(f"{OUT}/loan_accounts.csv", index=False)
print(f"loan:    {len(loan.head(50))} rows")

for name, file, id_col, extra in [
    ("bank",    "bank_statement.csv",   "account_id", {}),
    ("bureau",  "bureau_data.csv",      "account_id", {}),
    ("tele",    "telecall_history.csv", "account_id", {}),
    ("epfo",    "epfo_data.csv",        "account_id", {}),
]:
    df = pd.read_csv(f"{DATA}/{file}")
    sub = df[df[id_col].isin(ids)]
    out_file = f"{OUT}/{file}"
    sub.to_csv(out_file, index=False)
    print(f"{name:8}: {len(sub)} rows  ->  {out_file}")

# payment_history has LN prefix — convert to match
pay = pd.read_csv(f"{DATA}/payment_history.csv")
if 'ACCOUNT_NUMBER' in pay.columns:
    pay['ACCOUNT_NUMBER'] = pay['ACCOUNT_NUMBER'].str.replace('LN', 'ACC_')
    matched_ids = ids
    id_col = 'ACCOUNT_NUMBER'
else:
    matched_ids = ids
    id_col = 'account_id'
sub_pay = pay[pay[id_col].isin(matched_ids)]
sub_pay.to_csv(f"{OUT}/payment_history.csv", index=False)
print(f"payment: {len(sub_pay)} rows  ->  {OUT}/payment_history.csv")

print("\nSample data ready in data/sample/")
