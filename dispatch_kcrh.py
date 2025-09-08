import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import json
import base64
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score


data=pd.read_csv(r"C:\Users\admin\Desktop\Book1.csv", encoding='latin-1')
data.head()
#dropping the missing value
data.dropna(inplace=True)
data.isnull().sum()
#checking for duplicate values
data.duplicated().sum()

duplicate_rows = data[data.duplicated(keep=False)]

# Print the found duplicate rows.
if not duplicate_rows.empty:
    print("All instances of duplicate rows found in the dataset:")
    print(duplicate_rows.to_markdown(index=False, numalign='left', stralign='left'))
else:
    print("No exact duplicate rows were found.")

#dropping the duplicates
data.drop_duplicates(inplace=True)
data.duplicated().sum()

try:
    df = pd.read_csv('/content/Book1.csv', encoding='latin-1')
    print("Data loaded successfully.")
    print("Original data (first 5 rows):")
    print(df.head())
except FileNotFoundError:
    print("Error: 'Book1.csv' not found. Please upload the file to your Colab session.")
    exit()

# --- Step 2: Define the Priority Rules ---
# We will create a dictionary that maps specific diagnoses to a priority level.
# This mapping is based on the diagnoses found in your Book1.csv file.
priority_mapping = {
    # High Priority Diagnoses (for severe or acute conditions)
    "Road Traffic Injury": "High Priority",
    "Sepsis without septic shock": "High Priority",
    "Oral cellulitis or abscess": "High Priority",
    "Sepsis": "High Priority", # Added for completeness

    # Medium Priority Diagnoses (for serious but typically not critical conditions)
    "Malaria (suspected)": "Medium Priority",
    "Diarrhoea (Mild Dehydration)": "Medium Priority",
    "Acute upper respiratory infection (no pneumonia)": "Medium Priority",

    # Low Priority Diagnoses (for routine or less severe conditions)
    "Dental caries": "Low Priority",
    "Immunisation: Malaria": "Low Priority",
    "superficial injury of the hand and forearm": "Low Priority",
    "Diarrhoea (No Dehydration)": "Low Priority",
}

# --- Step 3: Create the New 'Case Priority' Column ---
# This function assigns a priority based on the diagnoses.
def get_priority(diagnoses):
    # Handle cases where diagnoses is NaN or not a string
    if not isinstance(diagnoses, str):
        return "Unknown Priority"

    # Split diagnoses if there are multiple, e.g., "Malaria (suspected), Sepsis without septic shock"
    diagnoses_list = [d.strip() for d in diagnoses.split(',')]

    # Check for the highest priority first to handle multiple diagnoses correctly
    if any(priority_mapping.get(d) == "High Priority" for d in diagnoses_list):
        return "High Priority"
    if any(priority_mapping.get(d) == "Medium Priority" for d in diagnoses_list):
        return "Medium Priority"
    if any(priority_mapping.get(d) == "Low Priority" for d in diagnoses_list):
        return "Low Priority"

    return "Unknown Priority"

# Apply the function to the 'Diagnoses' column to create the new 'Case Priority' column
df['Case Priority'] = df['Diagnoses'].apply(get_priority)

# --- Step 4: Display the Result ---
print("\nData with new 'Case Priority' column (first 5 rows):")
print(df[['Diagnoses', 'Case Priority']].head())

print("\nValue counts of the new 'Case Priority' column:")
print(df['Case Priority'].value_counts())

# Save the updated DataFrame to a new CSV file
output_filename = 'Book1_with_priority.csv'
df.to_csv(output_filename, index=False)

print(f"Updated dataset saved to '{output_filename}'")

Case_encoder = LabelEncoder()
df['Case Priority'] = Case_encoder.fit_transform(df['Case Priority'])
df.head()

location_encoder = LabelEncoder()
df['Location/Ward/Village'] = location_encoder.fit_transform(df['Location/Ward/Village'])
df.head()
Diagnoses_encoder = LabelEncoder()
df['Diagnoses'] = Diagnoses_encoder.fit_transform(df['Diagnoses'])
df.head()

Titles_encoder = LabelEncoder()
df['Investigation titles'] = Titles_encoder.fit_transform(df['Investigation titles'])
df.head()

Tests_encoder = LabelEncoder()
df['Investigation tests'] = Tests_encoder.fit_transform(df['Investigation tests'])
df.head()

TestResults_encoder = LabelEncoder()
df['Investigation test results'] = TestResults_encoder.fit_transform(df['Investigation test results'])
df.head()

Gender_encoder = LabelEncoder()
df['Gender'] = Gender_encoder.fit_transform(df['Gender'])
df.head()

#converting the age column to be in years only(Converting those in months)
def convert_age_to_years(age):
    if isinstance(age, str):
        if 'm' in age:
            # Assuming age with 'm' is in months, convert to years
            return int(age.replace('m', '')) / 12
        else:
            # Assuming age without 'm' is in years
            return int(age)
    # Handle cases where age is already a number or NaN if necessary
    return age

df['Age'] = df['Age'].apply(convert_age_to_years)
df.head()

#converting the visit date(monthname to string)
df['Visit date'] = pd.to_datetime(df['Visit date'], format='%d-%b-%y')
df.head()

#defining the dependent and independent variables
x = df.drop('Case Priority', axis=1)
y = df[['Case Priority']]

# Convert 'Visit date' to numerical timestamp
x['Visit date'] = x['Visit date'].astype('int64') // 10**9

#splitting the dataset into 70% training and 30% testing set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
print("y_train as DataFrame shape:", y_train.shape)
print("y_test as DataFrame shape:", y_test.shape)
print("x_train shape:",x_train.shape)
print("x_test shape:",x_test.shape)
#training and predicting decision tree model
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt_pred=dt.predict(x_test)
dt_pred

#evaluating decision tree
accuracy = accuracy_score(y_test, dt_pred)
f1 = f1_score(y_test, dt_pred, average='weighted')
recall = recall_score(y_test, dt_pred, average='weighted')
precision = precision_score(y_test, dt_pred, average='weighted')

dt_results=pd.DataFrame(['Decision Tree', accuracy, f1, recall, precision]).transpose()
dt_results.columns=['Model','Accuracy', 'F1 Score','Recall','Precision']
dt_results
filename = 'DecisionTree_model.pkl'
pickle.dump(dt, open(filename, 'wb'))

# Create a new LabelEncoder specifically for the 'Case Priority' column
case_priority_label_encoder = LabelEncoder()

# Load the data again to access the original 'Case Priority' column before encoding
# Assuming the file "Book1_with_priority.csv" saved earlier contains the 'Case Priority' column with string labels.
try:
    df_original_priority = pd.read_csv('Book1_with_priority.csv')
    print("Data with original 'Case Priority' labels loaded successfully.")
except FileNotFoundError:
    print("Error: 'Book1_with_priority.csv' not found. Please run the cell to save the updated data first.")
    exit()


# Fit the encoder on the original string labels of the 'Case Priority' column
case_priority_label_encoder.fit(df_original_priority['Case Priority'])

print("Label encoder for 'Case Priority' fitted on original labels.")
# You can check the classes learned by the encoder:
print("Classes:", case_priority_label_encoder.classes_)

# Example: predict for a new instance of Case Priority
# The input data should have the same features as the training data:
# 'Visit date', 'Gender', 'Age', 'Location/Ward/Village', 'Diagnoses',
# 'Investigation titles', 'Investigation tests', 'Investigation test results'
# All these features have been preprocessed into numerical format.

# Let's create a sample input data point. You'll need to provide values
# in the numerical format corresponding to your preprocessing steps.

# Example values (replace with actual values based on your data's encoding and scale):
# Visit date (numerical timestamp - example: a date in Feb 2024)
sample_visit_date = pd.to_datetime('2024-02-15').value // 10**9
# Gender (encoded: 0 for Female, 1 for Male)
sample_gender = 1 # Assuming Male
# Age (in years)
sample_age = 30.0
# Location/Ward/Village (encoded - example value)
sample_location = 598 # Replace with an actual encoded location value
# Diagnoses (encoded - example value)
sample_diagnoses = 76 # Replace with an actual encoded diagnoses value
# Investigation titles (encoded - example value)
sample_investigation_titles = 421 # Replace with an actual encoded value
# Investigation tests (encoded - example value)
sample_investigation_tests = 483 # Replace with an actual encoded value
# Investigation test results (encoded - example value)
sample_investigation_test_results = 3272 # Replace with an actual encoded value


# Combine the sample input data into a list in the correct order of features
input_data = [
    sample_visit_date,
    sample_gender,
    sample_age,
    sample_location,
    sample_diagnoses,
    sample_investigation_titles,
    sample_investigation_tests,
    sample_investigation_test_results
]
# Create a new LabelEncoder specifically for the 'Case Priority' column
case_priority_label_encoder = LabelEncoder()

# Load the data again to access the original 'Case Priority' column before encoding
# Assuming the file "Book1_with_priority.csv" saved earlier contains the 'Case Priority' column with string labels.
try:
    df_original_priority = pd.read_csv('Book1_with_priority.csv')
    print("Data with original 'Case Priority' labels loaded successfully.")
except FileNotFoundError:
    print("Error: 'Book1_with_priority.csv' not found. Please run the cell to save the updated data first.")
    exit()


# Fit the encoder on the original string labels of the 'Case Priority' column
case_priority_label_encoder.fit(df_original_priority['Case Priority'])

print("Label encoder for 'Case Priority' fitted on original labels.")
# You can check the classes learned by the encoder:
print("Classes:", case_priority_label_encoder.classes_)
# Changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for a single instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make the prediction using the trained Decision Tree model (dt)
prediction_encoded = dt.predict(input_data_reshaped)

# Print the predicted case priority (the output is the encoded priority)
print('Predicted Case Priority (encoded): {}'.format(prediction_encoded[0]))

# To get the actual case priority name, you would need the inverse mapping from the label encoder used for the 'Case Priority' column.
# Assuming you still have the 'label_encoder' object used for encoding the 'Case Priority' column (from cell WU000nwGLtLP):
# You might need to re-fit the label encoder on the original 'Case Priority' column from the df DataFrame
# to ensure it has the complete mapping if the original was overwritten or not stored.
# For demonstration, assuming the label_encoder object from cell WU000nwGLtLP is still available and contains the mapping:
try:
    # Use the new case_priority_label_encoder for inverse transform
    predicted_priority_label = case_priority_label_encoder.inverse_transform(prediction_encoded)
    print('Predicted Case Priority: {}'.format(predicted_priority_label[0]))
except NameError:
    print("Could not inverse transform: 'case_priority_label_encoder' not found.")
    print("Please run the cell to create and fit 'case_priority_label_encoder'.")
except ValueError as e:
    print(f"Error during inverse transform: {e}")
    print("This might happen if the encoder was not fitted on all possible predicted labels.")
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
    model_path = '/content/DecisionTree_model.pkl'
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
