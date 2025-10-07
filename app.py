# app.py - Intelligent Dispatch Dashboard (full, persistent, supports new + known patients)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import uuid
import sqlite3
import os
from typing import Optional
import io

# ---------------------------
# CONFIG / DB
# ---------------------------
# NOTE: using the DB filename you provided
DB_PATH = "dispatch_kcrh.py"
MODEL_FILE = "DecisionTree_model.pkl"
# Encoder file names expected (adjust if your files are named differently)
ENCODERS = {
    "Gender": "Gender_encoder.pkl",
    "Location/Ward/Village": "location_encoder.pkl",
    "Diagnoses": "Diagnoses_encoder.pkl",
    "Investigation titles": "Titles_encoder.pkl",
    "Investigation tests": "Tests_encoder.pkl",
    "Investigation test results": "TestResults_encoder.pkl",
    "Case Priority": "Case_encoder.pkl"
}

# ---------------------------
# DB INITIALIZATION & SCHEMA SAFETY
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

conn = init_db()

def ensure_table_and_columns():
    """
    Ensures the necessary tables and columns exist.
    This helps when an existing DB lacks the new columns we've added over time.
    """
    c = conn.cursor()
    # ambulances table (base)
    c.execute("""
        CREATE TABLE IF NOT EXISTS ambulances (
            id TEXT PRIMARY KEY,
            plate TEXT,
            driver TEXT,
            phone TEXT,
            status TEXT
        )
    """)
    # add optional columns for live location (text) and coordinates (lat/lon) if missing
    existing = [r[1] for r in c.execute("PRAGMA table_info('ambulances')").fetchall()]
    if "current_location" not in existing:
        try:
            c.execute("ALTER TABLE ambulances ADD COLUMN current_location TEXT")
        except Exception:
            pass
    if "latitude" not in existing:
        try:
            c.execute("ALTER TABLE ambulances ADD COLUMN latitude REAL")
        except Exception:
            pass
    if "longitude" not in existing:
        try:
            c.execute("ALTER TABLE ambulances ADD COLUMN longitude REAL")
        except Exception:
            pass

    # requests table (base)
    c.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id TEXT PRIMARY KEY,
            patient_name TEXT,
            patient_age INTEGER,
            patient_gender TEXT,
            visit_date TEXT,
            patient_location TEXT,
            patient_diagnosis TEXT,
            investigation_title TEXT,
            investigation_test TEXT,
            investigation_result TEXT,
            priority TEXT,
            status TEXT,
            ambulance_id TEXT,
            created_at TEXT,
            FOREIGN KEY(ambulance_id) REFERENCES ambulances(id)
        )
    """)
    # dispatch_logs table (centralized dispatch log) with columns we need
    c.execute("""
        CREATE TABLE IF NOT EXISTS dispatch_logs (
            id TEXT PRIMARY KEY,
            caller TEXT,
            caller_phone TEXT,
            request_location TEXT,
            request_reason TEXT,
            dispatch_location TEXT,
            request_id TEXT,
            vehicle_id TEXT,
            ambulance_plate TEXT,
            driver_name TEXT,
            driver_phone TEXT,
            status TEXT,
            requested_at TEXT,
            dispatched_at TEXT,
            enroute_at TEXT,
            at_scene_at TEXT,
            completed_at TEXT,
            response_time_seconds INTEGER,
            created_at TEXT,
            FOREIGN KEY(vehicle_id) REFERENCES ambulances(id),
            FOREIGN KEY(request_id) REFERENCES requests(id)
        )
    """)
    # ensure any newly introduced columns exist (for older DBs)
    dispatch_cols = [r[1] for r in c.execute("PRAGMA table_info('dispatch_logs')").fetchall()]
    needed = ["dispatch_location", "ambulance_plate", "driver_name", "driver_phone"]
    for col in needed:
        if col not in dispatch_cols:
            try:
                c.execute(f"ALTER TABLE dispatch_logs ADD COLUMN {col} TEXT")
            except Exception:
                pass

    conn.commit()

# Run schema safety once at startup
ensure_table_and_columns()

# ---------------------------
# Load ML model + encoders
# ---------------------------
loaded_model = None
label_encoders = {}

def load_model_and_encoders():
    global loaded_model, label_encoders
    # Load model
    if not os.path.exists(MODEL_FILE):
        st.error(f"Model file not found: {MODEL_FILE}. Put your trained model in the app folder.")
        st.stop()
    try:
        loaded_model = joblib.load(MODEL_FILE)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # Load encoders
    for key, fname in ENCODERS.items():
        if not os.path.exists(fname):
            st.error(f"Encoder file not found: {fname}. Required encoder: {key}")
            st.stop()
        try:
            label_encoders[key] = joblib.load(fname)
        except Exception as e:
            st.error(f"Failed to load encoder {fname}: {e}")
            st.stop()

# Call loader (stops app with clear error if missing)
load_model_and_encoders()

# ---------------------------
# Utility functions
# ---------------------------
def safe_encode(encoder, value):
    """Return encoded int if value in encoder.classes_, else None"""
    try:
        if value in getattr(encoder, "classes_", []):
            return int(encoder.transform([value])[0])
        else:
            return None
    except Exception:
        return None

def predict_priority(input_data: dict) -> str:
    """
    Attempt to predict priority. If any categorical value is unseen, return "Unknown Priority".
    input_data keys expected:
     - Visit date (date or str)
     - Gender, Age,
     - Location/Ward/Village, Diagnoses, Investigation titles, Investigation tests, Investigation test results
    """
    try:
        # convert visit date to timestamp
        visit_dt = pd.to_datetime(input_data["Visit date"])
        ts = visit_dt.timestamp()

        g = safe_encode(label_encoders["Gender"], input_data["Gender"])
        loc = safe_encode(label_encoders["Location/Ward/Village"], input_data["Location/Ward/Village"])
        diag = safe_encode(label_encoders["Diagnoses"], input_data["Diagnoses"])
        title = safe_encode(label_encoders["Investigation titles"], input_data["Investigation titles"])
        test = safe_encode(label_encoders["Investigation tests"], input_data["Investigation tests"])
        result = safe_encode(label_encoders["Investigation test results"], input_data["Investigation test results"])

        # if any encoding is None => unseen value; don't try to predict reliably
        if None in [g, loc, diag, title, test, result]:
            return "Unknown Priority"

        features = np.array([ts, g, int(input_data["Age"]), loc, diag, title, test, result]).reshape(1, -1)
        pred_num = loaded_model.predict(features)[0]
        pred_label = label_encoders["Case Priority"].inverse_transform([pred_num])[0]
        return pred_label
    except Exception as e:
        # log error to user and fallback
        st.warning(f"Prediction failed: {e}")
        return "Unknown Priority"

# ---------------------------
# Persistence helpers
# ---------------------------
def add_ambulance_db(plate: str, driver: str, phone: str, status: str = "available", current_location: str = None, latitude: Optional[float]=None, longitude: Optional[float]=None):
    aid = str(uuid.uuid4())
    c = conn.cursor()
    c.execute("INSERT INTO ambulances (id, plate, driver, phone, status, current_location, latitude, longitude) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (aid, plate, driver, phone, status, current_location, latitude, longitude))
    conn.commit()
    return aid

def update_ambulance_db(aid: str, plate: str, driver: str, phone: str, status: str, current_location: Optional[str]=None, latitude: Optional[float]=None, longitude: Optional[float]=None):
    c = conn.cursor()
    # Update only provided columns to avoid overwriting nulls unintentionally
    c.execute("UPDATE ambulances SET plate=?, driver=?, phone=?, status=? WHERE id=?", (plate, driver, phone, status, aid))
    if current_location is not None:
        c.execute("UPDATE ambulances SET current_location=? WHERE id=?", (current_location, aid))
    if latitude is not None:
        c.execute("UPDATE ambulances SET latitude=? WHERE id=?", (latitude, aid))
    if longitude is not None:
        c.execute("UPDATE ambulances SET longitude=? WHERE id=?", (longitude, aid))
    conn.commit()

def delete_ambulance_db(aid: str):
    c = conn.cursor()
    # remove ambulance assignments first (set ambulance_id null)
    c.execute("UPDATE requests SET ambulance_id=NULL WHERE ambulance_id=?", (aid,))
    c.execute("DELETE FROM ambulances WHERE id=?", (aid,))
    conn.commit()

def list_ambulances_df():
    return pd.read_sql("SELECT * FROM ambulances ORDER BY plate", conn)

def set_ambulance_status_db(aid: str, status: str):
    c = conn.cursor()
    c.execute("UPDATE ambulances SET status=? WHERE id=?", (status, aid))
    conn.commit()

def add_request_db(r: dict):
    c = conn.cursor()
    rid = r.get("id", str(uuid.uuid4()))
    c.execute("""
        INSERT INTO requests (id, patient_name, patient_age, patient_gender, visit_date,
        patient_location, patient_diagnosis, investigation_title, investigation_test, investigation_result,
        priority, status, ambulance_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        rid, r.get("patient_name"), r.get("patient_age"), r.get("patient_gender"), r.get("visit_date"),
        r.get("patient_location"), r.get("patient_diagnosis"), r.get("investigation_title"), r.get("investigation_test"), r.get("investigation_result"),
        r.get("priority"), r.get("status"), r.get("ambulance_id"), r.get("created_at")
    ))
    conn.commit()
    return rid

def update_request_db_ambulance(rid: str, aid: Optional[str], status: str):
    c = conn.cursor()
    c.execute("UPDATE requests SET ambulance_id=?, status=? WHERE id=?", (aid, status, rid))
    conn.commit()

def list_requests_df():
    return pd.read_sql("SELECT r.*, a.plate AS amb_plate, a.driver AS amb_driver, a.phone AS amb_phone FROM requests r LEFT JOIN ambulances a ON r.ambulance_id=a.id ORDER BY created_at DESC", conn)

# Dispatch log helpers
def add_dispatch_log(caller: str, caller_phone: str, request_location: str, request_reason: str,
                     dispatch_location: str, request_id: Optional[str], vehicle_id: Optional[str],
                     ambulance_plate: Optional[str]=None, driver_name: Optional[str]=None, driver_phone: Optional[str]=None,
                     status: str = "Requested"):
    """Insert a dispatch log including text-only dispatch_location and vehicle info."""
    did = str(uuid.uuid4())
    now = datetime.datetime.now().isoformat()
    requested_at = now
    c = conn.cursor()
    c.execute("""
        INSERT INTO dispatch_logs (id, caller, caller_phone, request_location, request_reason,
                                   dispatch_location, request_id, vehicle_id, ambulance_plate, driver_name, driver_phone,
                                   status, requested_at, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (did, caller, caller_phone, request_location, request_reason, dispatch_location,
          request_id, vehicle_id, ambulance_plate, driver_name, driver_phone, status, requested_at, now))
    conn.commit()
    return did

def update_dispatch_status(did: str, new_status: str):
    # set timestamp for the status if not already set
    ts = datetime.datetime.now().isoformat()
    c = conn.cursor()
    col = None
    if new_status == "Dispatched":
        col = "dispatched_at"
    elif new_status == "En route":
        col = "enroute_at"
    elif new_status == "At scene":
        col = "at_scene_at"
    elif new_status == "Completed":
        col = "completed_at"
    # update status and timestamp column
    if col:
        c.execute(f"UPDATE dispatch_logs SET status=?, {col}=? WHERE id=?", (new_status, ts, did))
    else:
        c.execute("UPDATE dispatch_logs SET status=? WHERE id=?", (new_status, did))
    # if completed -> compute response_time_seconds (dispatched_at - requested_at)
    if new_status == "Completed":
        # fetch requested_at and dispatched_at
        row = c.execute("SELECT requested_at, dispatched_at FROM dispatch_logs WHERE id=?", (did,)).fetchone()
        if row and row[0] and row[1]:
            try:
                requested = pd.to_datetime(row[0])
                dispatched = pd.to_datetime(row[1])
                secs = int((dispatched - requested).total_seconds())
                c.execute("UPDATE dispatch_logs SET response_time_seconds=? WHERE id=?", (secs, did))
            except Exception:
                pass
    conn.commit()

def assign_vehicle_and_create_dispatch(request_id: str, vehicle_id: str, caller: str = "", caller_phone: str = "", reason: str = "", dispatch_location: str = ""):
    # update request and ambulance
    update_request_db_ambulance(request_id, vehicle_id, "Dispatched")
    set_ambulance_status_db(vehicle_id, "busy")
    # fetch ambulance details for logging
    amb_df = list_ambulances_df()
    amb_row = amb_df[amb_df["id"] == vehicle_id]
    if not amb_row.empty:
        plate = amb_row.iloc[0]["plate"]
        driver = amb_row.iloc[0]["driver"]
        phone = amb_row.iloc[0].get("phone", "")
    else:
        plate = None
        driver = None
        phone = None
    # create dispatch log linking to request (stores dispatch_location and ambulance details)
    did = add_dispatch_log(caller, caller_phone, request_location="", request_reason=reason,
                           dispatch_location=dispatch_location, request_id=request_id, vehicle_id=vehicle_id,
                           ambulance_plate=plate, driver_name=driver, driver_phone=phone, status="Dispatched")
    # set dispatched_at immediately
    update_dispatch_status(did, "Dispatched")
    return did

def list_dispatches_df():
    return pd.read_sql("SELECT d.*, a.plate as vehicle_plate, a.driver as vehicle_driver, a.phone as vehicle_phone FROM dispatch_logs d LEFT JOIN ambulances a ON d.vehicle_id=a.id ORDER BY created_at DESC", conn)

# ---------------------------
# Analytics helpers
# ---------------------------
def compute_response_time_kpis(dispatches: pd.DataFrame):
    df = dispatches.copy()
    # ensure datetime
    for col in ["requested_at", "dispatched_at", "enroute_at", "at_scene_at", "completed_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    # compute response_seconds when possible
    df['response_seconds'] = (df['dispatched_at'] - df['requested_at']).dt.total_seconds()
    overall_avg = df['response_seconds'].dropna().mean()
    median = df['response_seconds'].dropna().median()
    completed_count = df[df['status']=='Completed'].shape[0]
    pending_count = df[df['status']!='Completed'].shape[0]
    return {
        'average_response_seconds': int(overall_avg) if not np.isnan(overall_avg) else None,
        'median_response_seconds': int(median) if not np.isnan(median) else None,
        'completed_count': int(completed_count),
        'pending_count': int(pending_count)
    }

def utilization_metrics(dispatches: pd.DataFrame, period_start: Optional[pd.Timestamp]=None, period_end: Optional[pd.Timestamp]=None):
    df = dispatches.copy()
    df['requested_at'] = pd.to_datetime(df['requested_at'], errors='coerce')
    if period_start is not None:
        df = df[df['requested_at'] >= period_start]
    if period_end is not None:
        df = df[df['requested_at'] <= period_end]
    # trips per vehicle
    trips = df.groupby('vehicle_plate').size().reset_index(name='trips')
    # average response per vehicle
    df['dispatched_at'] = pd.to_datetime(df['dispatched_at'], errors='coerce')
    df['response_seconds'] = (df['dispatched_at'] - df['requested_at']).dt.total_seconds()
    avg_response = df.groupby('vehicle_plate')['response_seconds'].mean().reset_index(name='avg_response_seconds')
    merged = trips.merge(avg_response, on='vehicle_plate', how='left')
    return merged

# ---------------------------
# App UI
# ---------------------------
st.set_page_config(page_title="Intelligent Dispatch Dashboard", layout="wide", page_icon="🚑")
st.title("🚑 Intelligent Dispatch Dashboard")

menu = st.sidebar.radio("Menu", ["Home", "Incoming Request", "Dispatch Board", "Driver Updates", "Driver Location Tracker", "Ambulance Dashboard", "Dispatch Log", "About"])

# ---------- HOME ----------
if menu == "Home":
    st.header("Overview")
    reqs_df = list_requests_df()
    amb_df = list_ambulances_df()
    disp_df = list_dispatches_df()

    st.metric("Total requests", len(reqs_df))
    st.metric("Total ambulances", len(amb_df))
    st.metric("Total dispatches", len(disp_df))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Recent requests")
        st.dataframe(reqs_df.head(10))
    with col2:
        st.subheader("Ambulance status")
        st.dataframe(amb_df)

    st.markdown("---")
    st.subheader("Quick KPIs")
    kpis = compute_response_time_kpis(disp_df)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg response (s)", kpis.get('average_response_seconds') or "N/A")
    k2.metric("Median response (s)", kpis.get('median_response_seconds') or "N/A")
    k3.metric("Completed dispatches", kpis.get('completed_count'))
    k4.metric("Pending dispatches", kpis.get('pending_count'))

    # show recent completed notifications
    st.markdown("---")
    st.subheader("Recent completed dispatches (notifications)")
    recent_completed = disp_df[disp_df['status'] == 'Completed'].head(10)
    if not recent_completed.empty:
        st.dataframe(recent_completed[['id','ambulance_plate','driver_name','dispatch_location','request_reason','completed_at']], use_container_width=True)
    else:
        st.write("No completed dispatches yet.")

# ---------- INCOMING REQUEST ----------
elif menu == "Incoming Request":
    st.header("Submit Incoming Request (accepts new values)")

    # Build options for select+other fields using encoder classes
    gender_options = list(getattr(label_encoders["Gender"], "classes_", []))
    location_options = list(getattr(label_encoders["Location/Ward/Village"], "classes_", []))
    diagnosis_options = list(getattr(label_encoders["Diagnoses"], "classes_", []))
    title_options = list(getattr(label_encoders["Investigation titles"], "classes_", []))
    test_options = list(getattr(label_encoders["Investigation tests"], "classes_", []))
    result_options = list(getattr(label_encoders["Investigation test results"], "classes_", []))

    with st.form("request_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            patient_name = st.text_input("Patient name")
            patient_age = st.number_input("Age", min_value=0, max_value=120, value=30)
            patient_gender = st.selectbox("Gender", gender_options + ["Other"], index=0)
            visit_date = st.date_input("Visit date", datetime.date.today())
        with c2:
            # Location: allow select OR manual entry
            loc_choice = st.selectbox("Location/Ward/Village", ["(choose)"] + location_options + ["Other"])
            if loc_choice == "Other" or loc_choice == "(choose)":
                patient_location = st.text_input("Enter Location (manual)")
            else:
                patient_location = loc_choice

            diag_choice = st.selectbox("Primary Diagnosis", ["(choose)"] + diagnosis_options + ["Other"])
            if diag_choice == "Other" or diag_choice == "(choose)":
                patient_diagnosis = st.text_input("Enter Diagnosis (manual)")
            else:
                patient_diagnosis = diag_choice

            title_choice = st.selectbox("Investigation Title", ["(choose)"] + title_options + ["Other"])
            if title_choice == "Other" or title_choice == "(choose)":
                investigation_title = st.text_input("Investigation Title (manual)")
            else:
                investigation_title = title_choice

            test_choice = st.selectbox("Investigation Test", ["(choose)"] + test_options + ["Other"])
            if test_choice == "Other" or test_choice == "(choose)":
                investigation_test = st.text_input("Investigation Test (manual)")
            else:
                investigation_test = test_choice

            result_choice = st.selectbox("Investigation Test Result", ["(choose)"] + result_options + ["Other"])
            if result_choice == "Other" or result_choice == "(choose)":
                investigation_result = st.text_input("Investigation Result (manual)")
            else:
                investigation_result = result_choice

        submitted = st.form_submit_button("Submit Request")
        if submitted:
            # Normalize fields (if empty manual entries, set to empty string)
            patient_gender_val = patient_gender if patient_gender != "Other" else ""
            patient_location_val = (patient_location or "").strip()
            patient_diagnosis_val = (patient_diagnosis or "").strip()
            investigation_title_val = (investigation_title or "").strip()
            investigation_test_val = (investigation_test or "").strip()
            investigation_result_val = (investigation_result or "").strip()

            # Prepare input for prediction
            input_data = {
                "Visit date": visit_date,
                "Gender": patient_gender_val if patient_gender_val else "",
                "Age": int(patient_age),
                "Location/Ward/Village": patient_location_val,
                "Diagnoses": patient_diagnosis_val,
                "Investigation titles": investigation_title_val,
                "Investigation tests": investigation_test_val,
                "Investigation test results": investigation_result_val
            }

            # Predict
            priority = predict_priority(input_data)

            # Save request to DB (ambulance_id null initially)
            request_record = {
                "id": str(uuid.uuid4()),
                "patient_name": patient_name or "Unknown",
                "patient_age": int(patient_age),
                "patient_gender": patient_gender_val or "Unknown",
                "visit_date": str(visit_date),
                "patient_location": patient_location_val,
                "patient_diagnosis": patient_diagnosis_val,
                "investigation_title": investigation_title_val,
                "investigation_test": investigation_test_val,
                "investigation_result": investigation_result_val,
                "priority": priority,
                "status": "Pending",
                "ambulance_id": None,
                "created_at": datetime.datetime.now().isoformat()
            }

            add_request_db(request_record)
            st.success(f"Request saved. Predicted priority: {priority}")

# ---------- DISPATCH BOARD ----------
elif menu == "Dispatch Board":
    st.header("Dispatch Board - assign ambulance to request & quick dispatch actions")

    # Load requests and ambulances
    reqs_df = list_requests_df()
    amb_df = list_ambulances_df()

    if reqs_df.empty:
        st.info("No requests yet.")
    else:
        # show a filter (optional)
        status_filter = st.selectbox("Filter by request status", ["All", "Pending", "Dispatched", "Completed"], index=0)
        display_df = reqs_df if status_filter == "All" else reqs_df[reqs_df["status"] == status_filter]
        st.dataframe(display_df[["id","patient_name","patient_age","patient_diagnosis","priority","status","amb_plate","amb_driver","amb_phone","created_at"]], use_container_width=True)

        st.markdown("---")
        st.subheader("Assign / Update Requests")
        # Choose request to act on
        sel_req_id = st.selectbox("Select request ID", ["(choose)"] + display_df["id"].tolist())
        if sel_req_id and sel_req_id != "(choose)":
            row = display_df[display_df["id"] == sel_req_id].iloc[0]
            st.write(f"**{row['patient_name']}**  •  Priority: **{row['priority']}**  •  Status: **{row['status']}**")

            # If pending -> allow assign ambulance
            if row["status"] == "Pending":
                # list available ambulances
                ambs = amb_df[amb_df["status"] == "available"]
                if ambs.empty:
                    st.warning("No available ambulances right now.")
                else:
                    # show display string for each ambulance
                    amb_display_map = {f"{r.plate} - {r.driver} ({r.phone})": r.id for r in ambs.itertuples()}
                    chosen_display = st.selectbox("Choose ambulance", ["(choose)"] + list(amb_display_map.keys()))
                    caller = st.text_input("Caller name (optional)")
                    caller_phone = st.text_input("Caller phone (optional)")
                    reason = st.text_input("Reason / notes (optional)")
                    # NEW: text-only dispatch location input
                    dispatch_location = st.text_input("Dispatch location (text only, e.g., Ahero, Kisumu Central)")

                    if chosen_display and chosen_display != "(choose)":
                        if st.button("Dispatch to chosen ambulance"):
                            ambulance_id = amb_display_map[chosen_display]
                            # create dispatch + update (includes dispatch_location)
                            did = assign_vehicle_and_create_dispatch(sel_req_id, ambulance_id, caller=caller, caller_phone=caller_phone, reason=reason, dispatch_location=dispatch_location)
                            st.success(f"Dispatched {row['patient_name']} -> {chosen_display} (Dispatch ID: {did})")
                            st.rerun()

            elif row["status"] == "Dispatched":
                st.info("This request is already dispatched.")
                if pd.notna(row["amb_plate"]):
                    st.write(f"Assigned ambulance: **{row['amb_plate']} - {row['amb_driver']}**")
                    if pd.notna(row["amb_phone"]):
                        phone = row["amb_phone"]
                        # click-to-call link
                        st.markdown(f"Driver contact: [{phone}](tel:{phone})")
                if st.button("Mark as Completed"):
                    # free ambulance and set request completed
                    if pd.notna(row["ambulance_id"]):
                        set_ambulance_status_db(row["ambulance_id"], "available")
                    update_request_db_ambulance(sel_req_id, None, "Completed")
                    # also find dispatch log(s) for this request and mark completed
                    c = conn.cursor()
                    logs = c.execute("SELECT id FROM dispatch_logs WHERE request_id=? ORDER BY created_at DESC", (sel_req_id,)).fetchall()
                    if logs:
                        update_dispatch_status(logs[0][0], "Completed")
                    st.success("Marked Completed")
                    st.rerun()
            elif row["status"] == "Completed":
                st.success("Request already completed.")

    st.markdown("---")
    st.subheader("Quick Dispatch Status Update (by Dispatch ID)")
    dispatches = list_dispatches_df()
    if dispatches.empty:
        st.info("No dispatch logs yet.")
    else:
        sel_disp = st.selectbox("Select dispatch ID", ["(choose)"] + dispatches['id'].tolist())
        if sel_disp and sel_disp != "(choose)":
            drow = dispatches[dispatches['id']==sel_disp].iloc[0]
            st.write(f"Dispatch for: **{drow.get('request_reason') or drow.get('request_id')}** • Status: **{drow['status']}**")
            new_status = st.selectbox("Update status to", ["Requested", "Dispatched", "En route", "At scene", "Completed"]) 
            if st.button("Update dispatch status"):
                update_dispatch_status(sel_disp, new_status)
                # if Completed and vehicle set -> free vehicle
                if new_status == "Completed" and pd.notna(drow.get('vehicle_id')):
                    set_ambulance_status_db(drow['vehicle_id'], 'available')
                st.success("Dispatch status updated")
                st.rerun()

# ---------- DRIVER UPDATES ----------
elif menu == "Driver Updates":
    st.header("Driver Updates — mark trips completed")

    ambs = list_ambulances_df()
    if ambs.empty:
        st.info("No ambulances registered yet.")
    else:
        # driver selects their ambulance by plate (or dispatcher picks)
        plate_options = ambs['plate'].fillna("").tolist()
        plate_choice = st.selectbox("Select ambulance (plate)", ["(choose)"] + plate_options)
        if plate_choice and plate_choice != "(choose)":
            sel = ambs[ambs['plate'] == plate_choice].iloc[0]
            st.write(f"Driver: **{sel['driver']}**  •  Phone: **{sel.get('phone','')}**  •  Status: **{sel.get('status','')}**")
            # show assigned dispatches for this vehicle
            dispatches = list_dispatches_df()
            vehicle_dispatches = dispatches[dispatches['vehicle_plate'] == plate_choice]
            active = vehicle_dispatches[vehicle_dispatches['status'].isin(['Dispatched', 'En route'])]
            if active.empty:
                st.info("No active dispatches for this ambulance.")
            else:
                st.subheader("Active dispatches")
                for _, d in active.iterrows():
                    st.markdown(f"**Dispatch ID:** {d['id']}  •  Location: {d.get('dispatch_location','')}  •  Status: {d['status']}")
                    if st.button(f"✅ Mark {d['id']} Completed", key=f"complete_{d['id']}"):
                        # mark completed: update dispatch log, free ambulance, update request
                        update_dispatch_status(d['id'], "Completed")
                        if pd.notna(d.get('vehicle_id')):
                            set_ambulance_status_db(d['vehicle_id'], "available")
                        # also set request status to Completed
                        if pd.notna(d.get('request_id')):
                            update_request_db_ambulance(d['request_id'], None, "Completed")
                        st.success(f"Dispatch {d['id']} marked completed. Dispatcher notified.")
                        st.rerun()

# ---------- DRIVER LOCATION TRACKER ----------
elif menu == "Driver Location Tracker":
    st.header("Driver Location Tracker")

    ambs = list_ambulances_df()
    if ambs.empty:
        st.info("No ambulances registered yet.")
    else:
        st.subheader("Update your current location (drivers)")
        plate_options = ambs['plate'].fillna("").tolist()
        plate_choice = st.selectbox("Select your ambulance (plate)", ["(choose)"] + plate_options)
        if plate_choice and plate_choice != "(choose)":
            sel_row = ambs[ambs['plate'] == plate_choice].iloc[0]
            st.write(f"Driver: **{sel_row['driver']}**  •  Phone: **{sel_row.get('phone','')}**")
            location_text = st.text_input("Current location (text, e.g., 'Ahero', 'Kondele')", value=sel_row.get('current_location') or "")
            col1, col2 = st.columns(2)
            with col1:
                lat_val = st.text_input("Latitude (optional)", value=str(sel_row.get('latitude') or ""))
            with col2:
                lon_val = st.text_input("Longitude (optional)", value=str(sel_row.get('longitude') or ""))
            if st.button("Submit location update"):
                # parse lat/lon if provided
                lat = None
                lon = None
                try:
                    if lat_val.strip() != "":
                        lat = float(lat_val)
                    if lon_val.strip() != "":
                        lon = float(lon_val)
                except Exception:
                    st.warning("Latitude/Longitude must be numeric or left blank.")
                update_ambulance_db(sel_row['id'], sel_row['plate'], sel_row['driver'], sel_row.get('phone',''), sel_row.get('status','available'), current_location=location_text, latitude=lat, longitude=lon)
                st.success("Location updated.")
                st.rerun()

        st.markdown("---")
        st.subheader("Live driver locations (dispatcher view)")
        ambs = list_ambulances_df()
        # show table with current_location and coords
        cols_to_show = [c for c in ["plate","driver","phone","status","current_location","latitude","longitude"] if c in ambs.columns]
        st.dataframe(ambs[cols_to_show], use_container_width=True)

        # Show map if any ambulances have valid coordinates
        coords = ambs.dropna(subset=['latitude','longitude']) if 'latitude' in ambs.columns and 'longitude' in ambs.columns else pd.DataFrame()
        if not coords.empty:
            # build dataframe for st.map (lat, lon)
            map_df = coords[['latitude','longitude']].rename(columns={'latitude':'lat','longitude':'lon'})
            # attach small hover info - Streamlit's st.map doesn't support hover, so show table with details above
            st.map(map_df)
        else:
            st.info("No valid latitude/longitude coordinates available. Provide lat/lon when updating location to see map pins.")

# ---------- DISPATCH LOG (History / Filters / Export / Analytics) ----------
elif menu == "Dispatch Log":
    st.header("Dispatch Log — history, filtering, analytics and export")
    dispatches = list_dispatches_df()

    # Filters
    st.sidebar.header("Dispatch Log filters")
    date_min = st.sidebar.date_input("From", value=(datetime.date.today() - datetime.timedelta(days=30)))
    date_max = st.sidebar.date_input("To", value=datetime.date.today())
    vehicle_filter = st.sidebar.selectbox("Vehicle (plate)", options=["All"] + sorted(dispatches['vehicle_plate'].dropna().unique().tolist()) if not dispatches.empty else ["All"])
    status_filter = st.sidebar.selectbox("Status", options=["All"] + sorted(dispatches['status'].dropna().unique().tolist()) if not dispatches.empty else ["All"]) 

    if not dispatches.empty:
        dispatches['requested_at'] = pd.to_datetime(dispatches['requested_at'], errors='coerce')
        mask = (dispatches['requested_at'].dt.date >= date_min) & (dispatches['requested_at'].dt.date <= date_max)
        df_filtered = dispatches[mask]
        if vehicle_filter != "All":
            df_filtered = df_filtered[df_filtered['vehicle_plate']==vehicle_filter]
        if status_filter != "All":
            df_filtered = df_filtered[df_filtered['status']==status_filter]

        st.subheader("Filtered dispatch logs")
        # show key columns including dispatch_location and ambulance/driver details
        show_cols = ["id","vehicle_plate","ambulance_plate","driver_name","driver_phone","dispatch_location","request_reason","status","requested_at","dispatched_at","completed_at"]
        available_cols = [c for c in show_cols if c in df_filtered.columns]
        st.dataframe(df_filtered[available_cols], use_container_width=True)

        # Export options
        st.markdown("---")
        st.subheader("Export / Download")
        col1, col2 = st.columns(2)
        with col1:
            csv = df_filtered.to_csv(index=False)
            st.download_button("Download CSV", data=csv, file_name=f"dispatch_logs_{date_min}_{date_max}.csv", mime='text/csv')
        with col2:
            # Excel
            towrite = io.BytesIO()
            try:
                with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
                    df_filtered.to_excel(writer, index=False, sheet_name='dispatch_logs')
                    writer.save()
                towrite.seek(0)
                st.download_button("Download Excel", data=towrite, file_name=f"dispatch_logs_{date_min}_{date_max}.xlsx", mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            except Exception:
                # fallback to csv if excel writer not available
                st.warning("Excel export failed in this environment — using CSV instead.")
                st.download_button("Download CSV (fallback)", data=csv, file_name=f"dispatch_logs_{date_min}_{date_max}.csv", mime='text/csv')

        # Analytics
        st.markdown("---")
        st.subheader("Analytics & Utilization")
        kpis = compute_response_time_kpis(df_filtered)
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg response (s)", kpis.get('average_response_seconds') or "N/A")
        col2.metric("Median response (s)", kpis.get('median_response_seconds') or "N/A")
        col3.metric("Completed dispatches", kpis.get('completed_count'))

        st.subheader("Utilization - trips per vehicle & avg response")
        start_ts = pd.to_datetime(date_min)
        end_ts = pd.to_datetime(date_max) + pd.Timedelta(days=1)
        util = utilization_metrics(dispatches, period_start=start_ts, period_end=end_ts)
        if util.empty:
            st.info("No trips in selected period.")
        else:
            st.dataframe(util, use_container_width=True)
    else:
        st.info("No dispatch logs available yet.")

# ---------- AMBULANCE DASHBOARD ----------
elif menu == "Ambulance Dashboard":
    st.header("Ambulance Dashboard")
    amb_df = list_ambulances_df()
    st.dataframe(amb_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Add new ambulance")
    with st.form("add_amb_form", clear_on_submit=True):
        plate = st.text_input("Plate number")
        driver = st.text_input("Driver name")
        phone = st.text_input("Driver phone")
        submitted = st.form_submit_button("Add ambulance")
        if submitted:
            if not plate or not driver:
                st.error("Enter plate and driver name")
            else:
                add_ambulance_db(plate.strip(), driver.strip(), phone.strip())
                st.success("Ambulance added")
                st.rerun()


    st.markdown("---")
    st.subheader("Manage ambulances")
    ambs = list_ambulances_df()
    if ambs.empty:
        st.info("No ambulances yet.")
    else:
        selected_amb = st.selectbox("Select ambulance to edit", ["(choose)"] + ambs["id"].tolist())
        if selected_amb and selected_amb != "(choose)":
            r = ambs[ambs["id"] == selected_amb].iloc[0]
            col1, col2 = st.columns(2)
            with col1:
                plate_new = st.text_input("Plate", r["plate"])
                driver_new = st.text_input("Driver", r["driver"])
            with col2:
                phone_new = st.text_input("Phone", r["phone"])
                status_new = st.selectbox("Status", ["available", "busy"], index=0 if r["status"] == "available" else 1)
            # optional fields
            loc_new = st.text_input("Current location (optional)", value=r.get("current_location") or "")
            lat_new = st.text_input("Latitude (optional)", value=str(r.get("latitude") or '') if 'latitude' in r.index else "")
            lon_new = st.text_input("Longitude (optional)", value=str(r.get("longitude") or '') if 'longitude' in r.index else "")

            if st.button("Update ambulance"):
                # attempt to parse coords
                lat_val = None
                lon_val = None
                try:
                    if lat_new.strip() != "":
                        lat_val = float(lat_new)
                    if lon_new.strip() != "":
                        lon_val = float(lon_new)
                except Exception:
                    st.warning("Latitude/Longitude must be numeric or left blank.")
                update_ambulance_db(selected_amb, plate_new, driver_new, phone_new, status_new, current_location=loc_new, latitude=lat_val, longitude=lon_val)
                st.success("Ambulance updated")
                st.rerun()
            if st.button("Delete ambulance"):
                delete_ambulance_db(selected_amb)
                st.success("Ambulance deleted")
                st.rerun()

# ---------- ABOUT ----------
elif menu == "About":
    st.header("About")
    st.write("""
    This Intelligent Dispatch Dashboard:
    - Loads a trained Decision Tree model and label encoders (expected .pkl files in the folder).
    - Accepts both known (encoder) categories and new free-text values for incoming patients.
    - If incoming categorical values are unseen the app falls back to 'Unknown Priority' so requests are still accepted.
    - Persists ambulances, requests and dispatch logs in a local SQLite DB so data survives restarts.
    - Lets you assign specific ambulance + driver (with phone) to each request and click-to-call the driver.

    New features in this version:
    - Dispatch location (text-only) is saved for each dispatch.
    - Dispatcher & driver features: drivers can mark trips complete (Driver Updates tab).
    - Driver Location Tracker: drivers update current location (text + optional lat/lon). Dispatcher sees a table and map when coordinates exist.
    - Dispatch logs include ambulance plate, driver name and phone at time of dispatch.
    - Schema safety: app auto-adds missing columns (so older DBs don't break).
    """)

# ---------------------------
# End
# ---------------------------



    


