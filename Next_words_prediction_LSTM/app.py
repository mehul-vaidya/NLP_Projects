"""
#conda create -p venv python==3.10.12
#conda activate [venv]
#pip install -r requirements.txt
#streamlit run app.py
# torch==1.12.1
"""

import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


from tensorflow.keras.models import model_from_json

with open("next_word_Prediction.json", "r") as json_file:
  model = json_file.read()
model = model_from_json(model)

tokenizer=None
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_sequence_len=16 #comes from training data
def predict_top_five_words(model, tokenizer, seed_text):
    print("call")
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    top_five_indexes = np.argsort(predicted[0])[::-1][:5]
    top_five_words = []
    for index in top_five_indexes:
        for word, idx in tokenizer.word_index.items():
            if idx == index:
                top_five_words.append(word)
                break
    return top_five_words



def main():
    
    st.title('Predict next 5 words')  

    #take input from user
    with st.form("Form 1",clear_on_submit=True):
      input_text = st.text_input('Enter Sentence (less than 16 words)')
      s_state=st.form_submit_button('Submit')
      if s_state:
        if(input_text is None or len(input_text)==0):
           st.error("Please enter data")
        else:   
          st.success("Entered statement is : " + str(input_text))
          output=predict_top_five_words(model ,tokenizer ,input_text )
          st.success(output)

          
if __name__ == '__main__':
    main()
    

