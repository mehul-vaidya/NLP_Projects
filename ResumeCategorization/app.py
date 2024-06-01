"""
#conda create -p venv python==3.9
#conda activate [venv]
#pip install -r requirements.txt
#streamlit run app.py
"""
import numpy as np
import streamlit as st
import pickle

KNN_Classifier = pickle.load(open('KNN_Classifier.pkl','rb'))
vectorizor = pickle.load(open('vectorizor.pkl','rb'))

import re
def preprocessing(txt):
    preprocess = re.sub('http\S+\s', ' ', txt)
    preprocess = re.sub('RT|cc', ' ', preprocess)
    preprocess = re.sub('#\S+\s', ' ', preprocess)
    preprocess = re.sub('@\S+', '  ', preprocess)
    preprocess = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', preprocess)
    preprocess = re.sub(r'[^\x00-\x7f]', ' ', preprocess)
    preprocess = re.sub('\s+', ' ', preprocess)
    return preprocess


category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}  

def main():
    
    st.title('Resume Category Prediction Web App')  

    #take input from user
    with st.form("Form 1",clear_on_submit=True):
      resume_text = st.text_input('Copy Resume Text  Here')
      s_state=st.form_submit_button('Predict Resume Category')
      if s_state:
        if(resume_text is None or len(resume_text)==0):
           st.error("Please enter data")
        else:   
          preprocessed_resume = preprocessing(resume_text)
          input_features = vectorizor.transform([preprocessed_resume])
          prediction_id = KNN_Classifier.predict(input_features)[0]
          category_name = category_mapping.get(prediction_id, "Unknown Category")

          if(category_name != "Unknown Category"):
            st.success("Resume Category is " + category_name) 
          else:
            st.error("Category can not be predicted")   
  
if __name__ == '__main__':
    main()
    

