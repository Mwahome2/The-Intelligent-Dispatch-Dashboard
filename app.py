import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import uuid
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# Load Model & Label Encoders
# ----------------------------
loaded_model = None
label_encoders = {}
try:
    model_path = 'DecisionTree_model.pkl'
    loaded_model = joblib.load(model_path)
    st.write(f"Loaded model from: {model_path}")

    label_encoders['Gender'] = joblib.load('Gender_encoder.pkl')
    label_encoders['Location/Ward/Village'] = joblib.load('location_encoder.pkl')
    label_encoders['Diagnoses'] = joblib.load('Diagnoses_encoder.pkl')
    label_encoders['Investigation titles'] = joblib.load('Titles_encoder.pkl')
    label_encoders['Investigation tests'] = joblib.load('Tests_encoder.pkl')
    label_encoders['Investigation test results'] = joblib.load('TestResults_encoder.pkl')
    label_encoders['Case Priority'] = joblib.load('Case_encoder.pkl')

    st.success("✅ Model and encoders loaded successfully")
except FileNotFoundError as e:
    st.error(f"Missing file: {e}. Please run training first.")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error loading assets: {e}")
    st.stop()

# ----------------------------
# Prediction Function
# ----------------------------
def predict_priority(input_data):
    try:
        processed_input = []
        processed_input.append(pd.to_datetime(input_data['Visit date']).timestamp())
        processed_input.append(label_encoders['Gender'].transform([input_data['Gender']])[0])
        processed_input.append(input_data['Age'])

        # Handle unseen/new values
        def safe_transform(encoder, value):
            if value in encoder.classes_:
                return encoder.transform([value])[0]
            else:
                return 0  # fallback to first class

        processed_input.append(safe_transform(label_encoders['Location/Ward/Village'], input_data['Location/Ward/Village']))
        processed_input.append(safe_transform(label_encoders['Diagnoses'], input_data['Diagnoses']))
        processed_input.append(safe_transform(label_encoders['Investigation titles'], input_data['Investigation titles']))
        processed_input.append(safe_transform(label_encoders['Investigation tests'], input_data['Investigation tests']))
        processed_input.append(safe_transform(label_encoders['Investigation test results'], input_data['Investigation test results']))

        input_data_reshaped = np.asarray(processed_input).reshape(1, -1)
        numerical_prediction = loaded_model.predict(input_data_reshaped)[0]
        predicted_priority = label_encoders['Case Priority'].inverse_transform([numerical_prediction])[0]
        return predicted_priority
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Unknown Priority"

# ----------------------------
# Ambulance Management
# ----------------------------
if 'ambulances' not in st.session_state:
    st.session_state.ambulances = [
        {"plate": "KDA 123A", "driver": "John Doe", "phone": "+254700111222", "status": "available"},
        {"plate": "KDB 456B", "driver": "Jane Smith", "phone": "+254733444555", "status": "available"},
        {"plate": "KDC 789C", "driver": "Ali Hassan", "phone": "+254722666777", "status": "available"}
    ]

def get_available_ambulances():
    return [f"{a['plate']} - {a['driver']} ({a['phone']})" for a in st.session_state.ambulances if a['status'] == "available"]

def mark_ambulance_status(plate, status):
    for a in st.session_state.ambulances:
        if plate.startswith(a['plate']):
            a['status'] = status

# ----------------------------
# Main Streamlit App
# ----------------------------
def main():
    st.set_page_config(page_title="Intelligent Dispatch Dashboard", layout="wide")
    st.title("🚑 Intelligent Dispatch Dashboard")
    st.markdown("---")

    if 'requests' not in st.session_state:
        st.session_state.requests = []

    nav = st.sidebar.radio("Navigation", ["Incoming Requests", "Dispatch Board", "Ambulances"])

    # ----------------------------
    # Incoming Requests
    # ----------------------------
    if nav == "Incoming Requests":
        st.header("1. Incoming Requests")
        with st.expander("Submit a New Request"):
            with st.form("new_request_form", clear_on_submit=True):
                col1, col2 = st.columns(2)
                with col1:
                    patient_name = st.text_input("Patient Name")
                    patient_gender = st.selectbox("Gender", ['M', 'F'])
                    patient_age = st.number_input("Patient Age (years)", min_value=0, max_value=120, value=30)
                    patient_visit_date = st.date_input("Visit Date", datetime.date.today())
                with col2:
                    patient_location = st.text_input("Patient Location/Ward/Village")
                    patient_diagnosis = st.text_input("Diagnosis")
                    patient_investigation_titles = st.text_input("Investigation Title")
                    patient_investigation_tests = st.text_input("Investigation Test")
                    patient_investigation_test_results = st.text_input("Investigation Test Results")

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
                        "patient_name": patient_name,
                        "patient_age": patient_age,
                        "patient_location": patient_location,
                        "patient_diagnosis": patient_diagnosis,
                        "priority": priority,
                        "status": "Pending"
                    }
                    st.session_state.requests.append(new_request)
                    st.success(f"✅ Request for {patient_name} submitted with '{priority}' priority!")

    # ----------------------------
    # Dispatch Board
    # ----------------------------
    elif nav == "Dispatch Board":
        st.header("2. Dispatch Board")
        if not st.session_state.requests:
            st.info("No active requests.")
        else:
            priority_order = {"High Priority": 0, "Medium Priority": 1, "Low Priority": 2, "Unknown Priority": 3}
            sorted_requests = sorted(st.session_state.requests, key=lambda x: priority_order.get(x['priority'], 4))

            for r in sorted_requests:
                cols = st.columns([2,2,2,2])
                cols[0].write(f"**{r['patient_name']}** ({r['patient_age']} yrs)")
                cols[1].write(f"Diagnosis: {r['patient_diagnosis']}")
                cols[2].write(f"Priority: {r['priority']}")
                cols[3].write(f"Status: {r['status']}")

                if r['status'] == "Pending":
                    available = get_available_ambulances()
                    if available:
                        amb = st.selectbox("Assign Ambulance", available, key=f"amb_{r['id']}")
                        if st.button("Dispatch", key=f"dispatch_{r['id']}"):
                            r['status'] = "Dispatched"
                            r['ambulance'] = amb
                            mark_ambulance_status(amb, "busy")
                            st.experimental_rerun()
                    else:
                        st.warning("⚠️ No ambulances available")

                elif r['status'] == "Dispatched":
                    st.write(f"🚑 Assigned Ambulance: {r['ambulance']}")
                    for a in st.session_state.ambulances:
                        formatted = f"{a['plate']} - {a['driver']} ({a['phone']})"
                        if formatted == r['ambulance']:
                            st.write(f"👨‍✈️ Driver: {a['driver']}")
                            st.write(f"📞 Contact: [{a['phone']}](tel:{a['phone']})")
                            break

                    if st.button("Complete", key=f"complete_{r['id']}"):
                        r['status'] = "Completed"
                        mark_ambulance_status(r['ambulance'], "available")
                        st.experimental_rerun()

                st.markdown("---")

    # ----------------------------
    # Ambulance Dashboard
    # ----------------------------
    elif nav == "Ambulances":
        st.header("3. Ambulance Dashboard")
        st.dataframe(pd.DataFrame(st.session_state.ambulances))

        with st.expander("➕ Add New Ambulance"):
            plate = st.text_input("Plate Number")
            driver = st.text_input("Driver Name")
            phone = st.text_input("Driver Phone")
            if st.button("Add Ambulance"):
                if plate and driver and phone:
                    st.session_state.ambulances.append(
                        {"plate": plate, "driver": driver, "phone": phone, "status": "available"}
                    )
                    st.success("🚑 Ambulance added successfully")

# ----------------------------
# Run App
# ----------------------------
if __name__ == '__main__':
    main()



