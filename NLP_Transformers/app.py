"""
#conda create -p venv python==3.10.0
#conda activate [venv]
#pip install -r requirements.txt
#streamlit run app.py
# torch==1.12.1
"""

import streamlit as st
from transformers import pipeline

def main():
    
    st.title('Analyse Input Text using Transformer Library')  

    #take input from user
    with st.form("Form 1",clear_on_submit=True):
      input_text = st.text_input('Enter Sentence')
      s_state=st.form_submit_button('Submit')
      if s_state:
        if(input_text is None or len(input_text)==0):
           st.error("Please enter data")
        else:   
          st.success("Entered statement is : " + str(input_text))

          #sentiment analysis
          clf = pipeline('sentiment-analysis')
          sentiment = clf(input_text)[0].get('label')
          sentiment_score = clf(input_text)[0].get('score')
          st.success("Sentiment is of type " + str(sentiment) + " with score " + str(sentiment_score)) 
          
          #text classification
          clf = pipeline("zero-shot-classification")
          result = clf(
            input_text,
            candidate_labels=["education", "politics", "business"],
          )
          labels =  result.get('labels')
          scores =  result.get('scores')
          for index in range(len(labels)):
             if(scores[index]>0.5):
                 st.success("Sentence belongs to topic " + labels[index])   
          
          #text generation
          gen = pipeline("text-generation", model="distilgpt2")
          st.success(gen(input_text)[0].get('generated_text'))

          
if __name__ == '__main__':
    main()
    

