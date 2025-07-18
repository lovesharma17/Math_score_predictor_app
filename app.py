import streamlit as st
import pickle as pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import numpy as np
import shap 
import matplotlib.pyplot as plt
from streamlit_extras.let_it_rain import rain

def get_data_clean():
    data = pd.read_csv('student-mat.csv' , sep=';')

    data.drop(['Mjob', 'Fjob', 'reason', 'guardian', 'nursery'], axis=1 , inplace=True)
   
    data = pd.get_dummies(data ,  columns=['school', 'sex', 'address', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 'higher', 'internet', 'romantic'], drop_first=True)
    
    le = LabelEncoder()
    data['famsize'] = le.fit_transform(data['famsize'])
    print(data)

    return data
    
def explainer(model , X):
   data = get_data_clean()
   X = data[['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'goout', 'G1', 'G2', 'absences',
                     'romantic_yes','address_U', 'paid_yes', 'higher_yes', 'internet_yes'  
                     ]]

   model = pickle.load(open("model.pkl", "rb"))
   explainer = shap.TreeExplainer(model)
   shap_values = explainer.shap_values(X)


   st.header('Feature Importance ♾️')


   fig1, ax1 = plt.subplots()
   plt.title('Feature importance based on SHAP values')
   shap.summary_plot(shap_values, X, show=False) 
   st.pyplot(fig1, bbox_inches='tight')
   plt.close(fig1) 

   st.write("---")


   fig2, ax2 = plt.subplots()
   plt.title("Feature importance based on SHAP values (Bar)")
   shap.summary_plot(shap_values, X, plot_type='bar', show=False) 
   st.pyplot(fig2, bbox_inches='tight')
   plt.close(fig2) 

def add_predictions(input_data):
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))

    
    feature_order = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'goout', 'G1', 'G2', 'absences',
                     'romantic_yes','address_U', 'paid_yes', 'higher_yes', 'internet_yes'  
                     ] 
    input_array = np.array([input_data[key] for key in feature_order]).reshape(1, -1)

    
    input_array_scaled = scaler.transform(input_array)


    predictions = model.predict(input_array_scaled)
    
    st.subheader("Math Score Prediction🧠")
    st.write(f"Predicted Score: {predictions[0]:.2f}")
    
    
    if predictions[0] > 15:
        st.write("Hurrah! You have passed the exam✨")
    else:
        st.write("You just need to work more😭")


    st.write("---")

def get_radar_chart(input_data):

    
   
    categories = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'goout','G1' , 'G2', 'absences', 'address_U', 'paid_yes', 'higher_yes', 'internet_yes', 'romantic_yes']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
      r=[
         input_data['age'], input_data['G1'], input_data['G2'], input_data['absences'], input_data['traveltime'], input_data['studytime'], input_data['failures'], input_data['goout'], input_data['romantic_yes'], input_data['paid_yes'], input_data['higher_yes'], input_data['address_U'], input_data['internet_yes']
      ],
      theta=categories,
      fill='toself',
      name='Input Value'
    ))

    fig.update_layout(
    polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 20]
    )),
    showlegend=True
    )

    fig.show()

    return fig
   
 


def add_sidebar():
    st.sidebar.header("Student Attributes")

    data = get_data_clean()

    slider_labels = [
        ("Age" , 'age'),
        ("Mother education" , 'Medu'),
        ("Fathers education" , 'Fedu'),
        ("Traveltime" , 'traveltime'),
        ("Studytime" , 'studytime'),
        ("Failure" , 'failures'),
        ("Going out" , 'goout'),
        ("G1 score" , 'G1'),
        ("G2 score" , 'G2'),
        ("Absences" , 'absences')
    ]

    selectbox_labels = [

        ("Romantic" , 'romantic_yes'),
        ("Address" , 'address_U'),
        ("Paid tutions" , "paid_yes"),
        ("Higher education" , "higher_yes"),
        ("Internet Access" , 'internet_yes')
    ]
    
    input_dict = {}

    for label , key in slider_labels :

     input_dict[key] = st.sidebar.slider(
            label,
            min_value=int(data[key].min()),
            max_value=int(data[key].max()),
            value=int(data[key].mean())
        )
    for label, key in selectbox_labels:
     input_dict[key] = 1 if st.sidebar.selectbox(label, key=key, options=("True", "False")) == "True" else 0
    
    return input_dict
        


def main():
    st.set_page_config(
        page_title="Math Exam Score Predictor",
        page_icon="math", 
        layout="wide",
        initial_sidebar_state= "expanded"
    )

    input_data = add_sidebar()

    with st.container():
        st.title("Math Exam Score Predictor🧮")
        rain(
        emoji="🎈",
        font_size=10,
        falling_speed=5,
        animation_length='infinite'
    )

        st.write("Please interact with this app to help check your math exam score. This app predicts using machine learning model the score of math using differnt attributes and conditions such as previous exam score , number of hours a student a studied and his attendence. You can update the measurement by hand using the sliders ")
    
    st.write("---")

    col1 , col2 =  st.columns({5,4})

    with col1 :
     radar_chart = get_radar_chart(input_data)
     st.plotly_chart(radar_chart)
    
  

    with col2:
       data = get_data_clean()
       X = data[['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'goout', 'G1', 'G2', 'absences',
                     'romantic_yes','address_U', 'paid_yes', 'higher_yes', 'internet_yes'  
                     ]]
       model = pickle.load(open("model.pkl", "rb"))
       
       add_predictions(input_data)
       explainer(model , X)

if __name__ == '__main__':
    main()