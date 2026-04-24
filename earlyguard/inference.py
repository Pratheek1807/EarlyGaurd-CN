"""
EarlyGuard — Inference Script
==============================
Give it 6 raw CSV files → get back risk scores, tiers,
intervention recommendations and cost-benefit per account.

Usage
-----
python inference.py \
    --loan       path/to/loan_accounts.csv \
    --bank       path/to/bank_statement.csv \
    --bureau     path/to/bureau_data.csv \
    --telecall   path/to/telecall_history.csv \
    --epfo       path/to/epfo_data.csv \
    --payment    path/to/payment_history.csv \
    --model_dir  path/to/model_folder/ \
    --output     earlyguard_predictions.csv
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb

# ============================================================
# FEATURE COLUMN ORDER — must match training exactly
# ============================================================
FEATURE_COLS = [
    'payment_deterioration_score',
    'financial_cushion_trend',
    'income_reliability_score',
    'bureau_stress_score',
    'contact_avoidance_score',
    'debt_pressure_ratio',
    'ptp_reliability_score',
    'sentiment_distress_signal',
    'recency_weighted_stress',
    'employment_event_recency',
]

# ============================================================
# UTILITIES
# ============================================================
def clamp(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))

def safe_div(a, b, default=0.0):
    return float(a / b) if (b and b != 0) else default

def linear_slope(values):
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=float)
    return float(np.polyfit(x, arr, 1)[0])

def last_n(df, date_col, n):
    return df.tail(n)

# ============================================================
# FEATURE EXTRACTORS  (same logic as feature_pipeline.py)
# ============================================================

def feat_payment_deterioration(pay_acc):
    last3 = last_n(pay_acc, 'payment_month', 3)
    ratio_avg = last3['partial_payment_ratio'].fillna(0.0).mean()
    missed    = (last3['payment_status'] == 'Missed').sum()
    days_late = last3['days_late'].fillna(30.0).mean()
    return clamp((1.0-ratio_avg)*0.40 + clamp(missed/3.0)*0.35 + clamp(days_late/30.0)*0.25)

def feat_financial_cushion_trend(bank_acc):
    last6 = last_n(bank_acc, 'statement_month', 6)
    last3 = last6.tail(3)
    slope_norm    = clamp(-linear_slope(last6['net_monthly_surplus'].fillna(0).values)/10_000)
    expense_stress= clamp(last3['expense_to_income_ratio'].fillna(1.0).mean()-0.5)
    near_zero     = clamp(last3['days_balance_near_zero'].fillna(0).sum()/30.0)
    return clamp(slope_norm*0.45 + expense_stress*0.30 + near_zero*0.25)

def feat_income_reliability(bank_acc, epfo_acc, emp):
    last3_bank = last_n(bank_acc, 'statement_month', 3)
    last6_epfo = last_n(epfo_acc, 'epfo_pull_date', 6)
    if emp == 'Salaried':
        delay_slope = linear_slope(last3_bank['salary_credit_delay_days'].fillna(0).values)
        delay_norm  = clamp(delay_slope / 5.0)
        zero_months = last6_epfo['zero_contribution_months_last_6m'].fillna(0).max()
        zero_norm   = clamp(zero_months / 6.0)
        job_change  = float(last6_epfo['job_change_detected'].fillna(False).any())
        return clamp(delay_norm*0.40 + zero_norm*0.35 + job_change*0.25)
    else:
        gst_map  = {'Filed':0.0,'Late Filed':0.5,'Not Filed':1.0}
        gst_stress = last6_epfo['gst_filing_status'].fillna('Not Filed').map(gst_map).fillna(1.0).mean()
        upi_map  = {'Stable':0.0,'Increasing':0.0,'Declining':1.0}
        upi_stress = last3_bank['business_upi_inflow_trend'].fillna('Declining').map(upi_map).fillna(1.0).mean()
        itr_stress = 0.0 if last6_epfo['itr_filed_flag'].fillna(False).any() else 1.0
        return clamp(gst_stress*0.40 + upi_stress*0.35 + itr_stress*0.25)

def feat_bureau_stress(bureau_acc):
    latest    = bureau_acc.sort_values('bureau_pull_date').iloc[-1]
    score_now = float(latest.get('credit_score_current',700) or 700)
    score_6m  = float(latest.get('credit_score_6m_ago', score_now) or score_now)
    drop_norm = clamp(-(score_now-score_6m)/100.0)
    missed    = clamp(float(latest.get('missed_payments_other_lenders_6m',0) or 0)/5.0)
    hard_30d  = clamp(float(latest.get('hard_enquiries_30d',0) or 0)/5.0)
    return clamp(drop_norm*0.40 + missed*0.35 + hard_30d*0.25)

def feat_contact_avoidance(tele_acc):
    last1 = last_n(tele_acc, 'contact_month', 1)
    resp  = float(last1['response_rate_30d'].fillna(0.0).mean())
    cons  = float(last1['consecutive_unanswered_count'].fillna(0).max())
    days  = float(last1['days_since_last_contact'].fillna(30).max())
    return clamp((1.0-clamp(resp))*0.35 + clamp(cons/10.0)*0.35 + clamp(days/30.0)*0.30)

def feat_debt_pressure(bank_acc, emi):
    last3     = last_n(bank_acc, 'statement_month', 3)
    income    = float(last3['total_monthly_inflow'].fillna(0).iloc[-1] if len(last3)>0 else 1.0)
    emi_other = float(last3['emi_outflow_other_lenders'].fillna(0).mean())
    return clamp(safe_div(emi+emi_other, income, 2.0), 0.0, 2.0)

def feat_ptp_reliability(tele_acc):
    last6 = last_n(tele_acc, 'contact_month', 6)
    last3 = last_n(tele_acc, 'contact_month', 3)
    if last6['ptp_made'].fillna(False).any():
        rate    = float(last6['ptp_fulfillment_rate'].fillna(0).iloc[-1])
        broken  = float(last3['ptp_broken_count_6m'].fillna(0).iloc[-1] if len(last3)>0 else 0)
        return clamp((1.0-clamp(rate))*0.50 + clamp(broken/3.0)*0.50)
    return 0.30

def feat_sentiment_distress(tele_acc):
    EMOTION = {'Cooperative':0.0,'Avoidant':0.4,'Frustrated':0.6,
                'Distressed':0.7,'Hostile':0.8,'Resigned':0.9}
    last1 = last_n(tele_acc, 'contact_month', 1)
    last3 = last_n(tele_acc, 'contact_month', 3)
    sent  = float(last1['sentiment_score_last_call'].fillna(0.5).mean())
    emo   = EMOTION.get(last1['dominant_emotion_last_call'].iloc[0] if len(last1)>0 else 'Avoidant', 0.4)
    hard  = float(last3['hardship_mentioned'].fillna(False).any())
    slope = clamp(1.0 - clamp(linear_slope(last3['sentiment_score_last_call'].fillna(0.5).values)+0.5))
    return clamp((1.0-clamp(sent))*0.35 + emo*0.30 + hard*0.20 + slope*0.15)

def feat_recency_weighted_stress(pay_acc):
    last6   = last_n(pay_acc, 'payment_month', 6)
    ratios  = last6['partial_payment_ratio'].fillna(0.0).values
    WEIGHTS = [0.05,0.08,0.12,0.17,0.25,0.33]
    n = len(ratios)
    if n == 0: return 0.5
    w = WEIGHTS[-n:]
    w = [wi/sum(w) for wi in w]
    return clamp(sum((1.0-r)*wi for r,wi in zip(ratios, w)))

def feat_employment_event_recency(epfo_acc, bank_acc, emp):
    last6_epfo = last_n(epfo_acc, 'epfo_pull_date', 6)
    last3_bank = last_n(bank_acc, 'statement_month', 3)
    if emp == 'Salaried':
        flags = last6_epfo['job_change_detected'].fillna(False).values
        if any(flags):
            months_since = int(np.argmax(flags[::-1]))
            return clamp(1.0 - months_since/6.0)
        zero  = float(last6_epfo['zero_contribution_months_last_6m'].fillna(0).max())
        days  = float(last6_epfo['days_since_last_contribution'].fillna(0).max())
        return 0.70 if zero>=3 else (0.50 if days>60 else 0.0)
    else:
        trend   = last3_bank['gst_turnover_trend'].fillna('Declining')
        n_dec   = int((trend=='Declining').sum())
        itr_ok  = last6_epfo['itr_filed_flag'].fillna(False).any()
        return 0.80 if n_dec>=2 else (0.50 if n_dec==1 else (0.40 if not itr_ok else 0.0))

# ============================================================
# PREDICTION
# ============================================================
def predict(art, X):
    Xs    = art['scaler'].transform(X)
    p_lr  = art['lr'].predict_proba(Xs)[:,1]
    p_rf  = art['rf'].predict_proba(X)[:,1]
    p_xgb = art['xgb'].predict(xgb.DMatrix(X))
    stack = np.column_stack([p_lr, p_rf, p_xgb])
    p_fin = art['meta'].predict_proba(stack)[:,1]
    return p_lr, p_rf, p_xgb, p_fin

def assign_tier(p30, p60):
    c = p30*0.60 + p60*0.40
    if c>=0.70: return 'Critical'
    if c>=0.45: return 'High'
    if c>=0.20: return 'Medium'
    return 'Low'

# ============================================================
# INTERVENTION & COST-BENEFIT
# ============================================================
COST_PER_CALL=45; COST_PER_SMS=2; COST_RESTRUCTURE=500; RECOVERY_RATE=0.15

def recommend_action(top_feat):
    MAP = {
        'payment_deterioration_score':'Immediate Payment Reminder Call',
        'recency_weighted_stress'    :'Urgent Payment Follow-up Call',
        'income_reliability_score'   :'Restructuring Conversation',
        'contact_avoidance_score'    :'WhatsApp Outreach — Non-Threatening',
        'bureau_stress_score'        :'Senior Agent Review',
        'employment_event_recency'   :'Hardship Assessment Call',
        'financial_cushion_trend'    :'Financial Health Check Call',
        'debt_pressure_ratio'        :'EMI Relief Discussion',
        'ptp_reliability_score'      :'Promise Commitment Re-engagement',
        'sentiment_distress_signal'  :'Empathy-First Outreach Call',
    }
    return MAP.get(top_feat,'Standard Follow-up Call')

def recommend_tone(sent_sig, hardship):
    if hardship or float(sent_sig or 0)>0.65: return 'Vulnerability — acknowledge before any ask'
    if float(sent_sig or 0)>0.40: return 'Empathetic — non-pressuring'
    return 'Professional — factual, solution-oriented'

def recommend_offer(dpr, fct, rws, emi):
    dpr=float(dpr or 0); fct=float(fct or 0); rws=float(rws or 0); emi=float(emi or 5000)
    if dpr>1.20:  return 'EMI Restructure — reduce EMI 30%, extend tenure'
    if fct>0.70:  return 'Payment Holiday — 1 month deferral, no penalty'
    if rws>0.75:  return f'Partial — accept ₹{round(emi*0.5/100)*100:,.0f} now, balance in 15 days'
    return f'Standard — full EMI ₹{emi:,.0f} by due date'

def cost_benefit(tier, p30, outstanding, emi):
    calls={'Low':0,'Medium':1,'High':3,'Critical':5}
    smss ={'Low':0,'Medium':2,'High':4,'Critical':6}
    cost = calls[tier]*COST_PER_CALL + smss[tier]*COST_PER_SMS + (COST_RESTRUCTURE if tier in ['High','Critical'] else 0)
    recovery = float(p30)*float(outstanding)*RECOVERY_RATE
    net = recovery - cost
    return round(cost,2), round(recovery,2), round(net,2), round(net/cost if cost else 0, 2)

# ============================================================
# MAIN PIPELINE
# ============================================================
def run_inference(loan_path, bank_path, bureau_path,
                  tele_path, epfo_path, pay_path,
                  model_dir, output_path):

    print("="*60)
    print("EARLYGUARD — INFERENCE PIPELINE")
    print("="*60)

    # ---- Load raw tables ----
    print("\nLoading raw CSV files...")
    loan   = pd.read_csv(loan_path)
    bank   = pd.read_csv(bank_path,   parse_dates=['statement_month'])
    bureau = pd.read_csv(bureau_path, parse_dates=['bureau_pull_date'])
    tele   = pd.read_csv(tele_path,   parse_dates=['contact_month'])
    epfo   = pd.read_csv(epfo_path,   parse_dates=['epfo_pull_date'])
    # Special handling for payment_history.csv which might have raw headers
    pay = pd.read_csv(pay_path)
    
    # Rename columns if they are in uppercase (ACCOUNT_NUMBER, PAYMENT_DATE, etc.)
    rename_map = {
        'ACCOUNT_NUMBER': 'account_id',
        'PAYMENT_DATE': 'payment_month',
        'PAYMENT_DUE_DATE': 'payment_due_date',
        'PAYMENT_AMOUNT': 'payment_amount',
        'EMI_AMOUNT': 'emi_amount',
        'PAYMENT_STATUS': 'payment_status'
    }
    pay = pay.rename(columns={c: rename_map[c] for c in rename_map if c in pay.columns})
    
    # Fix ID mismatch (LN00001 -> ACC_00001)
    if 'account_id' in pay.columns:
        pay['account_id'] = pay['account_id'].str.replace('LN', 'ACC_')

    
    # Parse dates explicitly (handling DD-MM-YYYY)
    for col in ['payment_month', 'payment_due_date']:
        if col in pay.columns:
            pay[col] = pd.to_datetime(pay[col], dayfirst=True, errors='coerce')
    
    # Calculate missing features if necessary
    if 'days_late' not in pay.columns and 'payment_month' in pay.columns and 'payment_due_date' in pay.columns:
        pay['days_late'] = (pay['payment_month'] - pay['payment_due_date']).dt.days
        pay['days_late'] = pay['days_late'].apply(lambda x: max(0, x) if pd.notnull(x) else 0)
        
    if 'partial_payment_ratio' not in pay.columns and 'payment_amount' in pay.columns and 'emi_amount' in pay.columns:
        pay['partial_payment_ratio'] = (pay['payment_amount'] / pay['emi_amount']).fillna(0.0)
        
    # Map status to 'Missed' if it's 'FAILED' (case-insensitive)
    if 'payment_status' in pay.columns:
        pay['payment_status'] = pay['payment_status'].apply(lambda x: 'Missed' if str(x).upper() == 'FAILED' else 'Paid')

    # Pre-sort for faster feature extraction
    print("  Pre-sorting data...")
    bank   = bank.sort_values(['account_id', 'statement_month'])
    bureau = bureau.sort_values(['account_id', 'bureau_pull_date'])
    tele   = tele.sort_values(['account_id', 'contact_month'])
    epfo   = epfo.sort_values(['account_id', 'epfo_pull_date'])
    pay    = pay.sort_values(['account_id', 'payment_month'])


    # Validate DPD = 0 constraint
    if 'dpd_at_snapshot' in loan.columns:
        invalid = loan[loan['dpd_at_snapshot'] > 0]
        if len(invalid) > 0:
            print(f"  WARNING: {len(invalid)} accounts have DPD > 0 — excluded (constraint)")
            loan = loan[loan['dpd_at_snapshot'] == 0]

    print(f"  Accounts to score: {len(loan)}")

    # ---- Load models ----
    print("\nLoading models...")
    with open(f'{model_dir}/model_30d.pkl','rb') as f: art30 = pickle.load(f)
    with open(f'{model_dir}/model_60d.pkl','rb') as f: art60 = pickle.load(f)
    print("  model_30d.pkl loaded")
    print("  model_60d.pkl loaded")

    # ---- Group tables ----
    bank_g   = {k:v for k,v in bank.groupby('account_id')}
    bureau_g = {k:v for k,v in bureau.groupby('account_id')}
    tele_g   = {k:v for k,v in tele.groupby('account_id')}
    epfo_g   = {k:v for k,v in epfo.groupby('account_id')}
    pay_g    = {k:v for k,v in pay.groupby('account_id')}

    # ---- Extract features ----
    print("\nExtracting features...")
    all_feats = []
    valid_accs = []
    skipped = 0

    for i, (_, acc) in enumerate(loan.iterrows()):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(loan)} accounts...")
        aid = acc['account_id']
        emp = acc['employment_type']
        emi = acc['emi_amount']

        if any(aid not in g for g in [bank_g,bureau_g,tele_g,epfo_g,pay_g]):
            skipped += 1
            continue

        # Extract features
        feats = {
            'payment_deterioration_score': feat_payment_deterioration(pay_g[aid]),
            'financial_cushion_trend'    : feat_financial_cushion_trend(bank_g[aid]),
            'income_reliability_score'   : feat_income_reliability(bank_g[aid], epfo_g[aid], emp),
            'bureau_stress_score'        : feat_bureau_stress(bureau_g[aid]),
            'contact_avoidance_score'    : feat_contact_avoidance(tele_g[aid]),
            'debt_pressure_ratio'        : feat_debt_pressure(bank_g[aid], emi),
            'ptp_reliability_score'      : feat_ptp_reliability(tele_g[aid]),
            'sentiment_distress_signal'  : feat_sentiment_distress(tele_g[aid]),
            'recency_weighted_stress'    : feat_recency_weighted_stress(pay_g[aid]),
            'employment_event_recency'   : feat_employment_event_recency(epfo_g[aid], bank_g[aid], emp),
        }
        all_feats.append(feats)
        valid_accs.append(acc)

    if not all_feats:
        print("  ERROR: No accounts found with data across all files.")
        return None

    # ---- Batch Predict ----
    print("\nRunning model predictions in batch...")
    X_batch = np.array([[f[c] for c in FEATURE_COLS] for f in all_feats])
    lr30, rf30, xgb30, p30 = predict(art30, X_batch)
    lr60, rf60, xgb60, p60 = predict(art60, X_batch)

    # ---- Post-process ----
    print("Generating recommendations and cost-benefit analysis...")
    rows = []
    for i, acc in enumerate(valid_accs):
        aid = acc['account_id']
        emp = acc['employment_type']
        emi = acc['emi_amount']
        feats = all_feats[i]
        
        prob30 = float(p30[i])
        prob60 = float(p60[i])
        tier = assign_tier(prob30, prob60)

        # SHAP top feature (from XGB weights proxy)
        feat_arr   = np.array([feats[c] for c in FEATURE_COLS])
        top_feat   = FEATURE_COLS[int(np.argmax(feat_arr))]

        # Intervention
        tele_last = last_n(tele_g[aid],'contact_month',1)
        sent_sig  = float(tele_last['sentiment_score_last_call'].fillna(0.5).mean())
        hardship  = bool(tele_last['hardship_mentioned'].fillna(False).any())
        best_hr   = int(tele_last['best_contact_hour'].fillna(10).iloc[0]) if len(tele_last)>0 else 10

        if tier in ['High','Critical']:
            action = recommend_action(top_feat)
            timing = f"{best_hr:02d}:00–{best_hr+1:02d}:00" + (
                     " | 1-2 days post salary credit" if emp=='Salaried'
                     else " | Mid-month")
            tone   = recommend_tone(sent_sig, hardship)
            offer  = recommend_offer(feats['debt_pressure_ratio'],
                                     feats['financial_cushion_trend'],
                                     feats['recency_weighted_stress'], emi)
        else:
            action=timing=tone=offer='N/A — Below intervention threshold'

        # Cost-benefit
        ci, cr, nb, roi = cost_benefit(tier, prob30, acc['outstanding_balance'], emi)

        rows.append({
            'account_id'                    : aid,
            'employment_type'               : emp,
            'product_type'                  : acc['product_type'],
            # Features
            **{f: round(feats[f],4) for f in FEATURE_COLS},
            # Probabilities
            'prob_30d_ensemble'             : round(prob30,4),
            'prob_60d_ensemble'             : round(prob60,4),
            'prob_30d_lr'                   : round(float(lr30[i]),4),
            'prob_30d_rf'                   : round(float(rf30[i]),4),
            'prob_30d_xgb'                  : round(float(xgb30[i]),4),
            # Risk tier & score (score = composite 0-100)
            'risk_tier'                     : tier,
            'risk_score'                    : round((prob30*0.60 + prob60*0.40)*100, 1),
            'top_signal'                    : top_feat,
            # Outstanding balance (dashboard column)
            'outstanding_balance_inr'       : float(acc['outstanding_balance']),
            # Current DPD (always 0 by constraint)
            'current_dpd'                   : int(acc.get('dpd_at_snapshot', 0)),
            # Loan type alias
            'loan_type'                     : acc['product_type'],
            # Intervention (dashboard-compatible names)
            'recommended_action'            : action,
            'intervention_timing'           : timing,
            'intervention_tone'             : tone,
            'intervention_offer'            : offer,
            # Cost-benefit (dashboard-compatible names)
            'intervention_cost_inr'         : ci,
            'expected_recovery_cost_inr'    : cr,
            'net_benefit_inr'               : nb,
            'roi'                           : roi,
            'intervention_justified'        : bool(nb>0),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, index=False)

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"  Accounts scored  : {len(df_out)}")
    print(f"  Accounts skipped : {skipped}")
    print(f"\n  Risk Tier Distribution:")
    for tier in ['Critical','High','Medium','Low']:
        n   = (df_out['risk_tier']==tier).sum()
        pct = n/len(df_out)*100 if len(df_out)>0 else 0
        bar = '#' * int(pct/5) + '-' * (20-int(pct/5))
        print(f"    {tier:<10} {bar} {n:>5} ({pct:.1f}%)")

    n_int = df_out['risk_tier'].isin(['High','Critical']).sum()
    print(f"\n  Accounts needing intervention : {n_int}")
    print(f"  Total intervention cost       : INR {df_out['intervention_cost_inr'].sum():,.0f}")
    print(f"  Total expected recovery saved : INR {df_out['expected_recovery_cost_inr'].sum():,.0f}")
    print(f"  Net benefit                   : INR {df_out['net_benefit_inr'].sum():,.0f}")
    print(f"\n  Output saved to: {output_path}")
    return df_out


# ============================================================
# CLI
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EarlyGuard Inference Pipeline')
    parser.add_argument('--loan',      required=True, help='Path to loan_accounts.csv')
    parser.add_argument('--bank',      required=True, help='Path to bank_statement.csv')
    parser.add_argument('--bureau',    required=True, help='Path to bureau_data.csv')
    parser.add_argument('--telecall',  required=True, help='Path to telecall_history.csv')
    parser.add_argument('--epfo',      required=True, help='Path to epfo_data.csv')
    parser.add_argument('--payment',   required=True, help='Path to payment_history.csv')
    parser.add_argument('--model_dir', required=True, help='Folder containing model_30d.pkl and model_60d.pkl')
    parser.add_argument('--output',    default='earlyguard_predictions.csv', help='Output CSV path')
    args = parser.parse_args()

    run_inference(
        loan_path   = args.loan,
        bank_path   = args.bank,
        bureau_path = args.bureau,
        tele_path   = args.telecall,
        epfo_path   = args.epfo,
        pay_path    = args.payment,
        model_dir   = args.model_dir,
        output_path = args.output,
    )
