"""
Creates a 1000-account test dataset sampled from the full data.
Samples randomly (not just head) for a representative spread.
Output: data/test/
"""
import pandas as pd
import os

DATA = "data"
OUT  = "data/test"
os.makedirs(OUT, exist_ok=True)

loan = pd.read_csv(f"{DATA}/loan_accounts.csv")
sample = loan.sample(n=1000, random_state=99)
ids = sample['account_id'].tolist()
sample.to_csv(f"{OUT}/loan_accounts.csv", index=False)
print(f"loan:    {len(sample)} rows")

for name, file, id_col in [
    ("bank",    "bank_statement.csv",   "account_id"),
    ("bureau",  "bureau_data.csv",      "account_id"),
    ("tele",    "telecall_history.csv", "account_id"),
    ("epfo",    "epfo_data.csv",        "account_id"),
]:
    df = pd.read_csv(f"{DATA}/{file}")
    sub = df[df[id_col].isin(ids)]
    sub.to_csv(f"{OUT}/{file}", index=False)
    print(f"{name:8}: {len(sub)} rows")

pay = pd.read_csv(f"{DATA}/payment_history.csv")
id_col = 'ACCOUNT_NUMBER' if 'ACCOUNT_NUMBER' in pay.columns else 'account_id'
pay_ids = [i.replace('ACC_', 'LN') for i in ids] if id_col == 'ACCOUNT_NUMBER' else ids
sub_pay = pay[pay[id_col].isin(pay_ids)]
sub_pay.to_csv(f"{OUT}/payment_history.csv", index=False)
print(f"payment: {len(sub_pay)} rows")

print(f"\nTest data ready in {OUT}/")
