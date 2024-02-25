import streamlit as st
from streamlit_option_menu import option_menu
import ultralytics
import pickle
from PIL import Image
import os
st.set_page_config(
        page_title="DiagnoML",
        page_icon="🩺",
   
)

diabetes_model=pickle.load(open("./saved_models/Diabetic_model.sav","rb"))
heart_disease_model=pickle.load(open("./saved_models/heart_disease_model.sav","rb"))
parkinsons_model=pickle.load(open("./saved_models/parkinsons_model.sav","rb"))
bone_fracture_model=ultralytics.YOLO(r"Saved_Models\best.pt")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with st.sidebar:
    selected=option_menu("DiagnoML - Disease Prediction App",
    ['Diabetes Disease Prediction',
    'Heart Disease Prediction',
    'Parkinsons Disease Prediction',
    "Bone Fracture Detection"],
    icons=['activity','heart','person','bandaid-fill'],
    default_index=0
    )

if(selected=="Diabetes Disease Prediction"):
    st.title("Diabetes Diseased Prediction Using ML")
        # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level (0-190)')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value (0-122)')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value (0-99)')
    
    with col2:
        Insulin = st.text_input('Insulin Level (0-846)')
    
    with col3:
        BMI = st.text_input('BMI value (0-67.1)')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value (0.08-2.42)')
    
    with col2:
        Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = '''The person is diabetic'''

        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex(0=f 1=m)')
        
    with col3:
        cp = st.text_input('Chest Pain types (0-3)')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure (94-200)')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl (126-564)')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (0-1)')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results (0-2)')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved (71-202)')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina (0-1)')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise (0-6.2)')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment (0-2)')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy (0-4)')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect (0-3)')

    # code for Prediction
    heart_diagnosis = ''
    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):

        heart_prediction = heart_disease_model.predict([[int(age), int(sex), int(cp), int(trestbps), int(chol), int(fbs), int(restecg),int(thalach),int(exang),int(oldpeak),int(slope),int(ca),int(thal)]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if (selected == "Parkinsons Disease Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)

if (selected == "Bone Fracture Detection"):
    st.title("Bone Fracture Detection")
    upload=st.file_uploader("Upload a file",accept_multiple_files=False)
    print(type(upload))
    if upload:
        st.image(upload,caption="Before Analyzing Image",use_column_width=True)
        target=Image.open(upload)
        target= target.convert("RGB")
        bone_fracture_model.predict(source=target,save=True,project="./results",hide_labels=True,hide_conf=True)
        predicted_image=os.listdir("./results/predict/")[0]
        predicted_image_=Image.open("./results/predict/"+predicted_image)
        st.image(predicted_image_,width=200,caption="After Analyzing Image",use_column_width=True)
        os.remove("./results/predict/"+str(predicted_image))
        os.rmdir("./results/predict/")