"""
EarlyGuard — Backend Server
============================
Serves the dashboard and runs the full inference pipeline
when raw CSVs are uploaded from the browser.

Usage
-----
cd earlyguard
python server.py
# Open http://localhost:5000
"""

import os, sys, json, uuid, threading, tempfile, io
from flask import Flask, request, jsonify, send_file, abort
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from inference import run_inference
from validation import run as run_validation
from survival_analysis import run as run_survival

app = Flask(__name__)

JOBS = {}
_lock = threading.Lock()


def _job(job_id, paths):
    tmpdir = paths["_tmpdir"]
    out_pred = os.path.join(tmpdir, "predictions.csv")
    out_val  = os.path.join(tmpdir, "validation_metrics.json")
    out_surv = os.path.join(tmpdir, "survival_curves.json")

    def update(stage, msg):
        with _lock:
            JOBS[job_id].update({"stage": stage, "progress": msg})

    try:
        update("inference", "Running inference pipeline — scoring accounts…")
        df = run_inference(
            loan_path   = paths["loan"],
            bank_path   = paths["bank"],
            bureau_path = paths["bureau"],
            tele_path   = paths["telecall"],
            epfo_path   = paths["epfo"],
            pay_path    = paths["payment"],
            model_dir   = BASE,
            output_path = out_pred,
        )
        if df is None or len(df) == 0:
            raise RuntimeError("Inference returned no results — check that all account IDs match across files.")

        with _lock:
            JOBS[job_id]["accounts"] = len(df)
            JOBS[job_id]["pred_path"] = out_pred

        update("validation", f"Inference done ({len(df):,} accounts). Running model validation…")
        run_validation(out_pred, paths["loan"], out_val)
        with _lock:
            JOBS[job_id]["val_path"] = out_val

        update("survival", "Validation done. Running survival analysis…")
        run_survival(out_pred, paths["loan"], out_surv)
        with _lock:
            JOBS[job_id]["surv_path"] = out_surv

        with _lock:
            JOBS[job_id].update({
                "stage": "done",
                "progress": f"Complete — {len(df):,} accounts scored",
                "done": True,
            })

    except Exception as exc:
        with _lock:
            JOBS[job_id].update({"stage": "error", "error": str(exc), "done": True})


# ── Static files ───────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file(os.path.join(BASE, "dashboard.html"))

_PIPELINE_DATA = {"earlyguard_predictions.csv", "validation_metrics.json", "survival_curves.json", "test_predictions.csv", "test_validation_metrics.json"}

@app.route("/<path:filename>")
def static_file(filename):
    # Block pre-existing data files — dashboard must get data from pipeline API only
    if filename in _PIPELINE_DATA:
        abort(404)
    path = os.path.join(BASE, filename)
    if os.path.isfile(path):
        return send_file(path)
    abort(404)


# ── Pipeline API ───────────────────────────────────────────────────

EMPTY_SCHEMAS = {
    "bank":     "account_id,statement_month,salary_credit_delay_days,total_monthly_inflow,net_monthly_surplus,expense_to_income_ratio,days_balance_near_zero,emi_outflow_other_lenders,business_upi_inflow_trend,gst_turnover_trend",
    "bureau":   "account_id,bureau_pull_date,credit_score_current,credit_score_6m_ago,missed_payments_other_lenders_6m,hard_enquiries_30d",
    "telecall": "account_id,contact_month,response_rate_30d,consecutive_unanswered_count,days_since_last_contact,ptp_made,ptp_fulfillment_rate,ptp_broken_count_6m,sentiment_score_last_call,dominant_emotion_last_call,hardship_mentioned,best_contact_hour",
    "epfo":     "account_id,epfo_pull_date,zero_contribution_months_last_6m,job_change_detected,days_since_last_contribution,itr_filed_flag,gst_filing_status",
    "payment":  "account_id,payment_month,payment_due_date,payment_amount,emi_amount,payment_status,days_late,partial_payment_ratio",
}

@app.route("/api/upload", methods=["POST"])
def upload():
    if "loan" not in request.files:
        return jsonify({"error": "loan_accounts.csv is required"}), 400

    tmpdir = tempfile.mkdtemp()
    paths  = {"_tmpdir": tmpdir}

    # Save uploaded files; create empty CSVs for anything not provided
    for key in ["loan", "bank", "bureau", "telecall", "epfo", "payment"]:
        dest = os.path.join(tmpdir, f"{key}.csv")
        if key in request.files and request.files[key].filename:
            request.files[key].save(dest)
        else:
            with open(dest, "w") as f:
                f.write(EMPTY_SCHEMAS[key] + "\n")
        paths[key] = dest

    job_id = str(uuid.uuid4())[:8]
    with _lock:
        JOBS[job_id] = {"stage": "queued", "progress": "Starting…", "done": False, "error": None}

    threading.Thread(target=_job, args=(job_id, paths), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id):
    with _lock:
        job = dict(JOBS.get(job_id, {}))
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify({
        "stage":    job.get("stage"),
        "progress": job.get("progress"),
        "done":     job.get("done", False),
        "error":    job.get("error"),
        "accounts": job.get("accounts", 0),
    })


@app.route("/api/predictions/<job_id>")
def predictions(job_id):
    with _lock:
        path = JOBS.get(job_id, {}).get("pred_path")
    if not path:
        return jsonify({"error": "Not ready"}), 404
    return send_file(path, mimetype="text/csv", download_name="predictions.csv")


@app.route("/api/validation/<job_id>")
def validation(job_id):
    with _lock:
        path = JOBS.get(job_id, {}).get("val_path")
    if not path:
        return jsonify({"error": "Not ready"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/survival/<job_id>")
def survival(job_id):
    with _lock:
        path = JOBS.get(job_id, {}).get("surv_path")
    if not path:
        return jsonify({"error": "Not ready"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


if __name__ == "__main__":
    print("\n  EarlyGuard server →  http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
