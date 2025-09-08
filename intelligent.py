import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import uuid
from sklearn.preprocessing import LabelEncoder

# --- 1. Load Machine Learning Model and Label Encoders ---
loaded_model = None
label_encoders = {}
try:
    # Load the model
    model_path = 'DecisionTree_model.pkl'
    loaded_model = joblib.load(model_path)
    st.write(f"Loaded model from: {model_path}")

    # Load all the saved label encoders
    label_encoders['Gender'] = joblib.load('/content/Gender_encoder.pkl')
    label_encoders['Location/Ward/Village'] = joblib.load('/content/location_encoder.pkl')
    label_encoders['Diagnoses'] = joblib.load('/content/Diagnoses_encoder.pkl')
    label_encoders['Investigation titles'] = joblib.load('/content/Titles_encoder.pkl')
    label_encoders['Investigation tests'] = joblib.load('/content/Tests_encoder.pkl')
    label_encoders['Investigation test results'] = joblib.load('/content/TestResults_encoder.pkl')
    label_encoders['Case Priority'] = joblib.load('/content/Case_encoder.pkl')

    st.success("Machine learning model and label encoders loaded successfully. ✅")
except FileNotFoundError as e:
    st.error(f"Error: A required file was not found. Please run the training script first to generate all the necessary .pkl files. Missing file: {e} ❌")
    st.stop() # Stop the app from running further if files are missing
except Exception as e:
    st.error(f"An unexpected error occurred while loading assets: {e} ⚠️")
    st.stop()

# --- 2. Prediction Function ---
def predict_priority(input_data):
    if loaded_model is None or not all(encoder in label_encoders for encoder in ['Gender', 'Location/Ward/Village', 'Diagnoses', 'Investigation titles', 'Investigation tests', 'Investigation test results', 'Case Priority']):
        return "Model or encoders not available"

    try:
        # Preprocess the input data using the loaded encoders
        processed_input = []
        # Convert date to timestamp, making it a numeric feature
        processed_input.append(pd.to_datetime(input_data['Visit date']).timestamp())
        processed_input.append(label_encoders['Gender'].transform([input_data['Gender']])[0])
        processed_input.append(input_data['Age'])
        processed_input.append(label_encoders['Location/Ward/Village'].transform([input_data['Location/Ward/Village']])[0])
        processed_input.append(label_encoders['Diagnoses'].transform([input_data['Diagnoses']])[0])
        processed_input.append(label_encoders['Investigation titles'].transform([input_data['Investigation titles']])[0])
        processed_input.append(label_encoders['Investigation tests'].transform([input_data['Investigation tests']])[0])
        processed_input.append(label_encoders['Investigation test results'].transform([input_data['Investigation test results']])[0])
        
        # Reshape the array for the model
        input_data_reshaped = np.asarray(processed_input).reshape(1, -1)

        # Make the numerical prediction
        numerical_prediction = loaded_model.predict(input_data_reshaped)[0]

        # Inverse transform the numerical prediction to get the original priority label
        predicted_priority = label_encoders['Case Priority'].inverse_transform([numerical_prediction])[0]
        
        return predicted_priority
    except Exception as e:
        st.error(f"An error occurred during prediction: {e} 😞")
        return "Prediction error"

# --- 3. Main Streamlit App Function ---
def main():
    st.set_page_config(page_title="Intelligent Dispatch Dashboard", layout="wide")

    st.title("🚑 Intelligent Dispatch Dashboard")
    st.markdown("---")

    if 'requests' not in st.session_state:
        st.session_state.requests = []

    # --- Incoming Requests Section ---
    st.header("1. Incoming Requests")
    with st.expander("Submit a New Request"):
        with st.form("new_request_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                patient_name = st.text_input("Patient Name")
                patient_gender = st.selectbox("Gender", ['M', 'F'])
                patient_age = st.number_input("Patient Age (in years)", min_value=0, max_value=120, value=30)
                patient_visit_date = st.date_input("Visit Date", datetime.date.today())
            with col2:
                # Use the classes from the loaded encoders for selectbox options
                patient_location_options = list(label_encoders['Location/Ward/Village'].classes_)
                patient_location = st.selectbox("Patient Location/Ward/Village", patient_location_options)
                
                diagnoses_options = list(label_encoders['Diagnoses'].classes_)
                patient_diagnosis = st.selectbox("Diagnoses", diagnoses_options)
                
                titles_options = list(label_encoders['Investigation titles'].classes_)
                patient_investigation_titles = st.selectbox("Investigation Titles", titles_options)
                
                tests_options = list(label_encoders['Investigation tests'].classes_)
                patient_investigation_tests = st.selectbox("Investigation Tests", tests_options)
                
                results_options = list(label_encoders['Investigation test results'].classes_)
                patient_investigation_test_results = st.selectbox("Investigation Test Results", results_options)
                
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

                if "Error" in priority or "Unknown" in priority or "not available" in priority:
                    st.error("Could not predict priority for this diagnosis. Please check model and encoder files.")
                else:
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
                    st.success(f"Request for {patient_name} submitted successfully with '{priority}' priority! 🌟")

    st.markdown("---")

    # --- Live Dispatch Board Section ---
    st.header("2. Live Dispatch Board")
    if not st.session_state.requests:
        st.info("No active requests. The dispatch board is clear. 😌")
    else:
        priority_order = {"High Priority": 0, "Medium Priority": 1, "Low Priority": 2, "Unknown Priority": 3}
        sorted_requests = sorted(st.session_state.requests, key=lambda x: priority_order.get(x['priority'], 4))

        for request in sorted_requests:
            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

            with col1:
                st.subheader(request['patient_name'])
                st.write(f"Age: {request['patient_age']}")
            with col2:
                st.write(f"**Diagnosis:**")
                st.write(request['patient_diagnosis'])
            with col3:
                st.write(f"**Priority:**")
                if request['priority'] == "High Priority":
                    st.error(request['priority'], icon="🚨")
                elif request['priority'] == "Medium Priority":
                    st.warning(request['priority'], icon="⚠️")
                elif request['priority'] == "Low Priority":
                    st.success(request['priority'], icon="✅")
                else:
                    st.info(request['priority'], icon="❓")
            with col4:
                st.write(f"**Status:**")
                st.write(request['status'])
            with col5:
                if request['status'] == "Pending":
                    if st.button("Dispatch", key=f"dispatch_{request['id']}"):
                        request['status'] = "Dispatched"
                        st.rerun()
                elif request['status'] == "Dispatched":
                    if st.button("Complete", key=f"complete_{request['id']}"):
                        request['status'] = "Completed"
                        st.rerun()
            st.markdown("---")

# Run the main function
if __name__ == '__main__':
    main()
