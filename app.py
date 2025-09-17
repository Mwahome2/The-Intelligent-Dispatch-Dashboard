# app.py - Intelligent Dispatch Dashboard with Ambulance Dashboard
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

    # load encoders (file names kept as in your original app)
    label_encoders['Gender'] = joblib.load('Gender_encoder.pkl')
    label_encoders['Location/Ward/Village'] = joblib.load('location_encoder.pkl')
    label_encoders['Diagnoses'] = joblib.load('Diagnoses_encoder.pkl')
    label_encoders['Investigation titles'] = joblib.load('Titles_encoder.pkl')
    label_encoders['Investigation tests'] = joblib.load('Tests_encoder.pkl')
    label_encoders['Investigation test results'] = joblib.load('TestResults_encoder.pkl')
    label_encoders['Case Priority'] = joblib.load('Case_encoder.pkl')

    st.success("✅ ML model and encoders loaded successfully")
except FileNotFoundError as e:
    st.error(f"❌ Missing file: {e} — please run training to produce the required .pkl files.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Unexpected error loading model/encoders: {e}")
    st.stop()

# ----------------------------
# 2) Prediction function
# ----------------------------
def predict_priority(input_data):
    """
    input_data: dict with keys:
      'Visit date', 'Gender', 'Age', 'Location/Ward/Village',
      'Diagnoses', 'Investigation titles', 'Investigation tests', 'Investigation test results'
    """
    try:
        # Build processed input in same order used during training
        processed_input = []
        # ensure Visit date is datetime-like -> timestamp
        visit_dt = pd.to_datetime(input_data['Visit date'])
        processed_input.append(visit_dt.timestamp())

        processed_input.append(label_encoders['Gender'].transform([input_data['Gender']])[0])
        processed_input.append(int(input_data['Age']))
        processed_input.append(label_encoders['Location/Ward/Village'].transform([input_data['Location/Ward/Village']])[0])
        processed_input.append(label_encoders['Diagnoses'].transform([input_data['Diagnoses']])[0])
        processed_input.append(label_encoders['Investigation titles'].transform([input_data['Investigation titles']])[0])
        processed_input.append(label_encoders['Investigation tests'].transform([input_data['Investigation tests']])[0])
        processed_input.append(label_encoders['Investigation test results'].transform([input_data['Investigation test results']])[0])

        input_array = np.asarray(processed_input).reshape(1, -1)
        num_pred = loaded_model.predict(input_array)[0]
        pred_label = label_encoders['Case Priority'].inverse_transform([num_pred])[0]
        return pred_label
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Unknown Priority"

# ----------------------------
# 3) App Initialization
# ----------------------------
st.set_page_config(page_title="Intelligent Dispatch Dashboard", layout="wide", page_icon="🚑")

# Initialize session state containers (preserve across reruns)
if 'requests' not in st.session_state:
    st.session_state.requests = []  # list of request dicts

if 'ambulances' not in st.session_state:
    # each ambulance: {"id": str, "plate": str, "driver": str, "status": "available" or "busy"}
    st.session_state.ambulances = []

# ----------------------------
# 4) Helper utilities
# ----------------------------
def add_ambulance(plate, driver):
    new_amb = {"id": str(uuid.uuid4()), "plate": plate.strip(), "driver": driver.strip(), "status": "available"}
    st.session_state.ambulances.append(new_amb)
    return new_amb

def find_ambulance_by_display(display):
    # display format: "PLATE - Driver"
    for a in st.session_state.ambulances:
        if f"{a['plate']} - {a['driver']}" == display:
            return a
    return None

def get_available_ambulance_displays():
    return [f"{a['plate']} - {a['driver']}" for a in st.session_state.ambulances if a['status'] == "available"]

def mark_ambulance_status_by_display(display, new_status):
    amb = find_ambulance_by_display(display)
    if amb:
        amb['status'] = new_status
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
    st.write("This dashboard manages incoming patient requests and assigns them to specific ambulances and drivers.")
    st.markdown("---")
    st.subheader("Quick Stats")
    total_requests = len(st.session_state.requests)
    pending = sum(1 for r in st.session_state.requests if r['status'] == "Pending")
    dispatched = sum(1 for r in st.session_state.requests if r['status'] == "Dispatched")
    completed = sum(1 for r in st.session_state.requests if r['status'] == "Completed")
    total_amb = len(st.session_state.ambulances)
    available_amb = sum(1 for a in st.session_state.ambulances if a['status'] == "available")
    busy_amb = total_amb - available_amb

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Requests", total_requests)
    c2.metric("Pending", pending)
    c3.metric("Dispatched", dispatched)
    c4.metric("Completed", completed)
    st.write("")
    c5, c6 = st.columns(2)
    c5.metric("Total Ambulances", total_amb)
    c6.metric("Available / Busy", f"{available_amb} / {busy_amb}")

# ----------------------------
# Incoming Requests
# ----------------------------
elif nav == "Incoming Requests":
    st.header("1. Incoming Requests")
    st.markdown("Submit a new patient request. Priority is predicted with the loaded ML model.")
    with st.expander("Submit a New Request", expanded=True):
        with st.form("new_request_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                patient_name = st.text_input("Patient Name")
                patient_gender = st.selectbox("Gender", ['M', 'F'])
                patient_age = st.number_input("Patient Age (years)", min_value=0, max_value=120, value=30)
                patient_visit_date = st.date_input("Visit Date", datetime.date.today())
            with col2:
                # Use classes from encoders for options (keeps compatibility)
                patient_location = st.selectbox("Location/Ward/Village", list(label_encoders['Location/Ward/Village'].classes_))
                patient_diagnosis = st.selectbox("Diagnosis", list(label_encoders['Diagnoses'].classes_))
                patient_investigation_titles = st.selectbox("Investigation Titles", list(label_encoders['Investigation titles'].classes_))
                patient_investigation_tests = st.selectbox("Investigation Tests", list(label_encoders['Investigation tests'].classes_))
                patient_investigation_test_results = st.selectbox("Investigation Test Results", list(label_encoders['Investigation test results'].classes_))

            submitted = st.form_submit_button("Submit Request")
            if submitted:
                input_data = {
                    'Visit date': patient_visit_date,
                    'Gender': patient_gender,
                    'Age': patient_age,
                    'Location/Ward/Village': patient_location,
                    'Diagnoses': patient_diagnosis,
                    'Investigation titles': patient_investigation_titles,
                    'Investigation tests': patient_investigation_tests,
                    'Investigation test results': patient_investigation_test_results
                }
                priority = predict_priority(input_data)

                new_request = {
                    "id": str(uuid.uuid4()),
                    "patient_name": patient_name.strip() or "Unknown",
                    "patient_age": int(patient_age),
                    "patient_location": patient_location,
                    "patient_diagnosis": patient_diagnosis,
                    "priority": priority,
                    "status": "Pending",
                    "ambulance": None,
                    "created_at": datetime.datetime.now().isoformat()
                }
                st.session_state.requests.append(new_request)
                st.success(f"Request submitted for {new_request['patient_name']} — Predicted priority: {priority}")

    # Show current requests summary with search/filter
    st.markdown("---")
    st.subheader("All Requests")
    if not st.session_state.requests:
        st.info("No requests yet.")
    else:
        df_requests = pd.DataFrame(st.session_state.requests)
        st.dataframe(df_requests[['id','patient_name','patient_age','patient_location','patient_diagnosis','priority','status','ambulance']], use_container_width=True)

# ----------------------------
# Dispatch Board
# ----------------------------
elif nav == "Dispatch Board":
    st.header("2. Live Dispatch Board")
    st.markdown("Assign a specific ambulance (plate + driver) to a request. Ambulance status will update automatically.")

    if not st.session_state.requests:
        st.info("No active requests.")
    else:
        # Sort requests by priority (High -> Medium -> Low -> Unknown)
        priority_order = {"High Priority": 0, "Medium Priority": 1, "Low Priority": 2, "Unknown Priority": 3, "Unknown": 3}
        sorted_requests = sorted(st.session_state.requests, key=lambda x: priority_order.get(x.get('priority','Unknown'), 4))

        for request in sorted_requests:
            # build layout per request
            col1, col2, col3, col4, col5, col6 = st.columns([1,1,2,1,1,1])
            with col1:
                st.subheader(request['patient_name'])
                st.write(f"Age: {request['patient_age']}")
                st.write(f"Location: {request.get('patient_location','')}")
            with col2:
                st.write("**Diagnosis:**")
                st.write(request['patient_diagnosis'])
            with col3:
                st.write("**Priority:**")
                p = request.get('priority', 'Unknown')
                if p == "High Priority":
                    st.error(p, icon="🚨")
                elif p == "Medium Priority":
                    st.warning(p, icon="⚠️")
                elif p == "Low Priority":
                    st.success(p, icon="✅")
                else:
                    st.info(p, icon="❓")
            with col4:
                st.write("**Status:**")
                st.write(request['status'])
            with col5:
                # Assign ambulance when Pending
                if request['status'] == "Pending":
                    available_displays = get_available_ambulance_displays()
                    if available_displays:
                        chosen_display = st.selectbox("Assign Ambulance", options=available_displays, key=f"amb_{request['id']}")
                        if st.button("Dispatch", key=f"dispatch_{request['id']}"):
                            # set request fields
                            request['status'] = "Dispatched"
                            request['ambulance'] = chosen_display
                            # mark ambulance busy
                            mark_ambulance_status_by_display(chosen_display, "busy")
                            st.success(f"Dispatched {request['patient_name']} -> {chosen_display}")
                            st.experimental_rerun()
                    else:
                        st.warning("No ambulances available")
                elif request['status'] == "Dispatched":
                    st.write(f"🚐 {request.get('ambulance','')}")
                    if st.button("Complete", key=f"complete_{request['id']}"):
                        request['status'] = "Completed"
                        # free the ambulance
                        if request.get('ambulance'):
                            mark_ambulance_status_by_display(request['ambulance'], "available")
                        st.success(f"Marked {request['patient_name']} as Completed")
                        st.experimental_rerun()
                elif request['status'] == "Completed":
                    st.write("✅ Completed")
            with col6:
                # show small details / timestamp
                st.caption(f"Created: {request.get('created_at','-')}")
            st.markdown("---")

# ----------------------------
# Ambulance Dashboard
# ----------------------------
elif nav == "Ambulance Dashboard":
    st.header("🚐 Ambulance Dashboard")
    st.markdown("Add ambulances, view status, and manually change status if needed.")

    # Add ambulance form
    with st.form("add_ambulance_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            plate = st.text_input("Plate Number")
        with col2:
            driver = st.text_input("Driver Name")
        add_submitted = st.form_submit_button("Add Ambulance")
        if add_submitted:
            if plate.strip() == "" or driver.strip() == "":
                st.error("Enter both plate number and driver name.")
            else:
                new_amb = add_ambulance(plate.strip(), driver.strip())
                st.success(f"Ambulance added: {new_amb['plate']} - {new_amb['driver']}")

    st.markdown("---")
    # List all ambulances
    if not st.session_state.ambulances:
        st.info("No ambulances added yet.")
    else:
        amb_df = pd.DataFrame(st.session_state.ambulances)
        # Show basic table
        st.dataframe(amb_df[['id','plate','driver','status']], use_container_width=True)

        st.subheader("Manage Ambulance Status")
        colA, colB = st.columns(2)
        with colA:
            pick = st.selectbox("Select Ambulance", options=[f"{a['plate']} - {a['driver']}" for a in st.session_state.ambulances], key="adm_pick")
        with colB:
            new_status = st.selectbox("Set Status", options=["available","busy"], key="adm_status")
            if st.button("Update Status"):
                changed = mark_ambulance_status_by_display(pick, new_status)
                if changed:
                    st.success(f"Ambulance {pick} status set to {new_status}")
                else:
                    st.error("Could not update ambulance status.")
                st.experimental_rerun()

# ----------------------------
# About
# ----------------------------
elif nav == "About":
    st.header("About this Dashboard")
    st.write("""
    This application:
    - Predicts priority for incoming requests using a pre-trained model.
    - Lets dispatchers assign a **specific ambulance and driver** to each dispatch.
    - Tracks ambulance availability (available / busy).
    - Frees ambulances when dispatch is completed.
    """)
    st.markdown("---")
    st.write("Files expected in the app directory:")
    st.write("- DecisionTree_model.pkl")
    st.write("- Gender_encoder.pkl, location_encoder.pkl, Diagnoses_encoder.pkl, Titles_encoder.pkl, Tests_encoder.pkl, TestResults_encoder.pkl, Case_encoder.pkl")
    st.write("")
    st.write("If any of those are missing the app will stop with an error message.")

# ----------------------------
# End
# ----------------------------
