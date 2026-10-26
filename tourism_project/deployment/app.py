import os
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

model_path = os.path.join(os.path.dirname(__file__), "best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction")
st.write("Fill the customer details below to predict if they'll purchase a travel package")

Age = st.slider("Age", 18, 70, 30)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.slider("Duration of Pitch (mins)", 0, 100, 15)
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female", "Others"])
NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 1, 5, 2)
NumberOfFollowups = st.slider("Number of Follow-ups", 1, 10, 3)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
NumberOfTrips = st.slider("Number of Trips", 1, 20, 3)
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.slider("Number of Children Visited", 0, 5, 1)
Designation = st.selectbox("Designation", ["Executive", "Manager", "AVP", "VP", "Sr. Manager"])
MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, value=30000.0)

input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])

classification_threshold = 0.45

if st.button("Predict"):
    probabilities = model.predict_proba(input_data)[0]
    prob = probabilities[1]
    pred = int(prob >= classification_threshold)
    result = "will purchase the travel package" if pred == 1 else "is unlikely to purchase"
    
    st.markdown(f"### Prediction: Customer **{result}**")
    st.markdown(f"Probability Score: **{prob*100:.2f}%**")
    
    # Plotly interactive chart rendering inside Streamlit
    classes = ['Will NOT Purchase (0)', 'Will Purchase (1)']
    colors = ['#EF553B', '#636EFA'] 
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=classes,
        y=probabilities,
        marker_color=colors,
        text=[f"{p*100:.2f}%" for p in probabilities],
        textposition='auto',
        hoverinfo='text'
    ))
    
    fig.add_shape(type="line",
        xref="paper", yref="y",
        x0=0, y0=classification_threshold, x1=1, y1=classification_threshold,
        line=dict(color="Green", width=3, dash="dash"),
    )
    
    fig.update_layout(
        title="Customer Purchase Probability Distribution",
        yaxis=dict(title="Probability Score", range=[0, 1]),
        xaxis=dict(title="Target Outcomes"),
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
