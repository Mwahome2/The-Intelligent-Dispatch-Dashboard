# app.py - Intelligent Dispatch Dashboard (accepts both known + new patient data)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import uuid
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# 1) Load ML model + encoders
# ----------------------------
loaded_model = None
label_encoders = {}

try:
    model_path = 'DecisionTree_model.pkl'
    loaded_model = joblib.load(model_path)

    # load encoders
    label_encoders['Gender'] = joblib.load('Gender_encoder.pkl')
    label_encoders['Location/Ward/Village'] = joblib.load('location_encoder.pkl')
    label_encoders['Diagnoses'] = joblib.load('Diagnoses_encoder.pkl')
    label_encoders['Investigation titles'] = joblib.load('Titles_encoder.pkl')
    label_encoders['Investigation tests'] = joblib.load('Tests_encoder.pkl')
    label_encoders['Investigation test results'] = joblib.load('TestResults_encoder.pkl')
    label_encoders['Case Priority'] = joblib.load('Case_encoder.pkl')

    st.success("✅ ML model and encoders loaded successfully")
except FileNotFoundError as e:
    st.error(f"❌ Missing file: {e}")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Unexpected error loading model/encoders: {e}")
    st.stop()

# ----------------------------
# 2) Prediction function
# ----------------------------
def safe_encode(encoder, value):
    """Try encoding a value; if unseen, return None"""
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return None

def predict_priority(input_data):
    try:
        processed_input = []
        visit_dt = pd.to_datetime(input_data['Visit date'])
        processed_input.append(visit_dt.timestamp())

        g = safe_encode(label_encoders['Gender'], input_data['Gender'])
        loc = safe_encode(label_encoders['Location/Ward/Village'], input_data['Location/Ward/Village'])
        diag = safe_encode(label_encoders['Diagnoses'], input_data['Diagnoses'])
        title = safe_encode(label_encoders['Investigation titles'], input_data['Investigation titles'])
        test = safe_encode(label_encoders['Investigation tests'], input_data['Investigation tests'])
        result = safe_encode(label_encoders['Investigation test results'], input_data['Investigation test results'])

        processed_input.extend([g if g is not None else 0,
                                int(input_data['Age']),
                                loc if loc is not None else 0,
                                diag if diag is not None else 0,
                                title if title is not None else 0,
                                test if test is not None else 0,
                                result if result is not None else 0])

        if None in [g, loc, diag, title, test, result]:
            return "Unknown Priority"

        arr = np.asarray(processed_input).reshape(1, -1)
        pred_num = loaded_model.predict(arr)[0]
        pred_label = label_encoders['Case Priority'].inverse_transform([pred_num])[0]
        return pred_label
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Unknown Priority"

# ----------------------------
# 3) App Initialization
# ----------------------------
st.set_page_config(page_title="Intelligent Dispatch Dashboard", layout="wide", page_icon="🚑")

if 'requests' not in st.session_state:
    st.session_state.requests = []
if 'ambulances' not in st.session_state:
    st.session_state.ambulances = []

# ----------------------------
# 4) Ambulance helpers
# ----------------------------
def add_ambulance(plate, driver):
    new_amb = {"id": str(uuid.uuid4()), "plate": plate.strip(), "driver": driver.strip(), "status": "available"}
    st.session_state.ambulances.append(new_amb)
    return new_amb

def get_available_ambulances():
    return [f"{a['plate']} - {a['driver']}" for a in st.session_state.ambulances if a['status'] == "available"]

def mark_ambulance_status(display, new_status):
    for a in st.session_state.ambulances:
        if f"{a['plate']} - {a['driver']}" == display:
            a['status'] = new_status
            return True
    return False

# ----------------------------
# 5) App Navigation
# ----------------------------
st.title("🚑 Intelligent Dispatch Dashboard")
nav = st.sidebar.radio("Navigation", ["Home", "Incoming Requests", "Dispatch Board", "Ambulance Dashboard", "About"])

# ----------------------------
# Home
# ----------------------------
if nav == "Home":
    st.header("Welcome")
    st.write("Manage patient requests, predict priority, and assign to ambulances + drivers.")
    total = len(st.session_state.requests)
    pending = sum(r['status']=="Pending" for r in st.session_state.requests)
    dispatched = sum(r['status']=="Dispatched" for r in st.session_state.requests)
    completed = sum(r['status']=="Completed" for r in st.session_state.requests)
    amb_total = len(st.session_state.ambulances)
    amb_avail = sum(a['status']=="available" for a in st.session_state.ambulances)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Requests", total)
    c2.metric("Pending", pending)
    c3.metric("Dispatched", dispatched)
    c4.metric("Completed", completed)
    c5.metric("Ambulances Available", amb_avail)

# ----------------------------
# Incoming Requests
# ----------------------------
elif nav == "Incoming Requests":
    st.header("Incoming Requests")
    with st.form("new_request_form", clear_on_submit=True):
        col1,col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Patient Name")
            patient_gender = st.selectbox("Gender", ['M','F','Other'])
            patient_age = st.number_input("Age", 0, 120, 30)
            patient_visit_date = st.date_input("Visit Date", datetime.date.today())
        with col2:
            # Known + Other for new values
            loc_choice = st.selectbox("Location/Ward/Village",
                list(label_encoders['Location/Ward/Village'].classes_) + ["Other"])
            if loc_choice=="Other":
                patient_location = st.text_input("Enter Location")
            else:
                patient_location = loc_choice

            diag_choice = st.selectbox("Diagnosis",
                list(label_encoders['Diagnoses'].classes_) + ["Other"])
            if diag_choice=="Other":
                patient_diagnosis = st.text_input("Enter Diagnosis")
            else:
                patient_diagnosis = diag_choice

            title_choice = st.selectbox("Investigation Title",
                list(label_encoders['Investigation titles'].classes_) + ["Other"])
            patient_investigation_titles = st.text_input("Enter Title") if title_choice=="Other" else title_choice

            test_choice = st.selectbox("Investigation Test",
                list(label_encoders['Investigation tests'].classes_) + ["Other"])
            patient_investigation_tests = st.text_input("Enter Test") if test_choice=="Other" else test_choice

            result_choice = st.selectbox("Investigation Result",
                list(label_encoders['Investigation test results'].classes_) + ["Other"])
            patient_investigation_test_results = st.text_input("Enter Result") if result_choice=="Other" else result_choice

        submitted = st.form_submit_button("Submit Request")
        if submitted:
            input_data = {
                "Visit date": patient_visit_date,
                "Gender": patient_gender,
                "Age": patient_age,
                "Location/Ward/Village": patient_location,
                "Diagnoses": patient_diagnosis,
                "Investigation titles": patient_investigation_titles,
                "Investigation tests": patient_investigation_tests,
                "Investigation test results": patient_investigation_test_results
            }
            priority = predict_priority(input_data)
            new_req = {
                "id": str(uuid.uuid4()),
                "patient_name": patient_name or "Unknown",
                "patient_age": int(patient_age),
                "patient_location": patient_location,
                "patient_diagnosis": patient_diagnosis,
                "priority": priority,
                "status": "Pending",
                "ambulance": None
            }
            st.session_state.requests.append(new_req)
            st.success(f"Added request for {new_req['patient_name']} with priority {priority}")

    if st.session_state.requests:
        st.subheader("All Requests")
        st.dataframe(pd.DataFrame(st.session_state.requests), use_container_width=True)

# ----------------------------
# Dispatch Board
# ----------------------------
elif nav == "Dispatch Board":
    st.header("Dispatch Board")
    if not st.session_state.requests:
        st.info("No requests yet")
    else:
        for r in st.session_state.requests:
            cols = st.columns([2,2,2,2])
            cols[0].write(f"**{r['patient_name']}** ({r['patient_age']} yrs)")
            cols[1].write(f"Diagnosis: {r['patient_diagnosis']}")
            cols[2].write(f"Priority: {r['priority']}")
            cols[3].write(f"Status: {r['status']}")
            if r['status']=="Pending":
                available = get_available_ambulances()
                if available:
                    amb = st.selectbox("Assign Ambulance", available, key=f"amb_{r['id']}")
                    if st.button("Dispatch", key=f"dispatch_{r['id']}"):
                        r['status']="Dispatched"
                        r['ambulance']=amb
                        mark_ambulance_status(amb,"busy")
                        st.experimental_rerun()
                else:
                    st.warning("No ambulances available")
            elif r['status']=="Dispatched":
                st.write(f"🚑 {r['ambulance']}")
                if st.button("Complete", key=f"complete_{r['id']}"):
                    r['status']="Completed"
                    mark_ambulance_status(r['ambulance'],"available")
                    st.experimental_rerun()
            st.markdown("---")

# ----------------------------
# Ambulance Dashboard
# ----------------------------
elif nav == "Ambulance Dashboard":
    st.header("Ambulance Dashboard")
    with st.form("add_amb", clear_on_submit=True):
        plate = st.text_input("Plate Number")
        driver = st.text_input("Driver Name")
        if st.form_submit_button("Add Ambulance"):
            if plate and driver:
                add_ambulance(plate,driver)
                st.success("Ambulance added")
    if st.session_state.ambulances:
        st.dataframe(pd.DataFrame(st.session_state.ambulances), use_container_width=True)

# ----------------------------
# About
# ----------------------------
elif nav == "About":
    st.header("About")
    st.write("""
    - Predicts patient priority (if values match training data).  
    - Accepts **new patients with unseen values** (priority becomes 'Unknown Priority').  
    - Assigns requests to **specific ambulances + drivers**.  
    - Tracks ambulance availability.  
    """)


# ----------------------------
# End
# ----------------------------
