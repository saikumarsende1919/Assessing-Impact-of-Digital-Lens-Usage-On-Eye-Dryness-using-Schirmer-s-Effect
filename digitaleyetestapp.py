import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Function to classify Schirmer's test results
def classify_schirmers_test(result):
    if result > 30:
        return "Reflex tearing Overactive/insufficient tear drainage"
    elif 16 <= result <= 30:
        return "Normal tear production"
    elif 11 <= result <= 15:
        return "Low normal tear production"
    elif 7 <= result <= 10:
        return "Borderline tear production"
    elif 2 <= result < 6:
        return "Abnormal Hypo Secretion"
    elif(result<=1) :
        return "Sjogren's Syndrome "
    

# Streamlit app
st.title("Schirmer's Eye Test")

# Import the dataset
df = pd.read_csv("digital-eye.csv")
df.drop(['Name'], axis=1, inplace=True)
# Specify numeric columns
numeric_columns = df.select_dtypes(include=['number']).columns

# Fill null values with the mean for numeric columns only
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
#drop duplicate values
df = df.drop_duplicates()
numeric_df = df.select_dtypes(include='number')

# Calculate the IQR for each numeric column
Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1

# Define outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Clean outliers by replacing them with the lower or upper bound
cleaned_df = numeric_df.clip(lower=lower_bound, upper=upper_bound, axis=1)

# Replace the numeric columns in the original DataFrame with the cleaned values
for column in cleaned_df.columns:
    df[column] = cleaned_df[column]
dd=numeric_df

# Splitting dataset into X and y where y has all the target variables
X = dd.drop(['Schimers1Lefteye', 'Schimers1righteye', 'Schimers2Lefteye', 'Schimers2righteye'], axis=1)
y = dd[['Schimers1Lefteye', 'Schimers1righteye', 'Schimers2Lefteye', 'Schimers2righteye']]


# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build and train the model
# Build and train the model
model = RandomForestRegressor(
    n_estimators=40,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=1,
    random_state=42  
)
model.fit(X_train, y_train)


# Evaluate the model
y_pred = model.predict(X_test)

# Show dataset
st.subheader("Dataset")
st.write(dd)

# Show model evaluation metrics
st.subheader("Model Evaluation Metrics")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"Model Used : RandomForestRegressor")
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"R-squared (R2) Score: {r2}")

# Form inputs
st.write('Please enter the following details:')
age = st.number_input('Age', min_value=0)
sex = st.selectbox('Sex', [1, 2])  # 1 represents Male, 2 represents Female
wearables = st.number_input('wearables', min_value=0)
duration = st.number_input('Duration', min_value=0)
online_platforms = st.number_input('onlineplatforms', min_value=0)
nature = st.number_input('Nature', min_value=0)
screen_illumination = st.number_input('screenillumination', min_value=0)
working_years = st.number_input('working Years', min_value=0)
hours_spent_daily_curricular = st.number_input('hoursspentdailycurricular', min_value=0)
hours_spent_daily_non_curricular = st.number_input('hoursspentdailynoncurricular', min_value=0)
gadgets_used = st.number_input('Gadgetsused', min_value=0)
level_of_gadget_with_respect_to_eyes = st.number_input('levelofgadgetwithrespecttoeyes', min_value=0)
distance_kept_between_eyes_and_gadget = st.number_input('Distancekeptbetweeneyesandgadget', min_value=0)
avg_nighttime_usage_per_day = st.number_input('Avgnighttimeusageperday', min_value=0)
blinking_during_screen_usage = st.number_input('Blinkingduringscreenusage', min_value=0)
difficulty_in_focusing_after_using_screens = st.number_input('Difficultyinfocusingafterusingscreens', min_value=0)
frequency_of_complaints = st.number_input('frequencyofcomplaints', min_value=0)
severity_of_complaints = st.number_input('Severityofcomplaints', min_value=0)
rvis = st.number_input('RVIS', min_value=0)
ocular_symptoms_observed_lately = st.number_input('Ocularsymptomsobservedlately', min_value=0)
symptoms_observing_at_least_half_of_the_times = st.number_input('Symptomsobservingatleasthalfofthetimes', min_value=0)
complaints_frequency = st.number_input('Complaintsfrequency', min_value=0)
frequency_of_dry_eyes = st.number_input('frequencyofdryeyes', min_value=0)

# Predict button
if st.button('Predict'):
    # Collect input data
    input_data = {
        'Age': age,
        'Sex': sex,
        'wearables': wearables,
        'Duration': duration,
        'onlineplatforms': online_platforms,
        'Nature': nature,
        'screenillumination': screen_illumination,
        'workingyears': working_years,
        'hoursspentdailycurricular': hours_spent_daily_curricular,
        'hoursspentdailynoncurricular': hours_spent_daily_non_curricular,
        'Gadgetsused': gadgets_used,
        'levelofgadjetwithrespecttoeyes': level_of_gadget_with_respect_to_eyes,
        'Distancekeptbetweeneyesandgadjet': distance_kept_between_eyes_and_gadget,
        'Avgnighttimeusageperday': avg_nighttime_usage_per_day,
        'Blinkingduringscreenusage': blinking_during_screen_usage,
        'Difficultyinfocusingafterusingscreens': difficulty_in_focusing_after_using_screens,
        'freqquencyofcomplaints': frequency_of_complaints,
        'Severityofcomplaints': severity_of_complaints,
        'RVIS': rvis,
        'Ocularsymptomsobservedlately': ocular_symptoms_observed_lately,
        'Symptomsobservingatleasthalfofthetimes': symptoms_observing_at_least_half_of_the_times,
        'Complaintsfrequency': complaints_frequency,
        'frequencyofdryeyes': frequency_of_dry_eyes
    }

    # Map sex input to 'Male' or 'Female'
    sex_mapping = {1: 'Male', 2: 'Female'}
    sex = sex_mapping[sex]

    # Create DataFrame from input data
    input_df = pd.DataFrame([input_data])

    # Replace column names to match the model
    input_df.columns = X_train.columns

    # Make prediction
    prediction = model.predict(input_df)

    # Display prediction
    st.write('Prediction Results:')
    for i, eye in enumerate(["Left", "Right"]):
        for j, schimer in enumerate(["Schimers1", "Schimers2"]):
            result = prediction[0][i * 2 + j]
            classification = classify_schirmers_test(result)
            st.write(f'{schimer}{eye}eye: {result} mm ({classification})')

