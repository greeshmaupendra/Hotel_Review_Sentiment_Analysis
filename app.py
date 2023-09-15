import numpy as np
import pandas as pd
import re
import streamlit as st
import warnings
from nltk.corpus import stopwords
import pickle
from afinn import Afinn
import time
warnings.filterwarnings("ignore")
import nltk
nltk.download('stopwords')
stopwords_set = set(stopwords.words('english'))
st.set_option('deprecation.showPyplotGlobalUse', False)
from annotated_text import annotated_text
from annotated_text import annotated_text, annotation



import base64
import streamlit as st
import plotly.express as px

df = px.data.iris()

#@st.experimental_memo
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("image1.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://images.unsplash.com/photo-1517840901100-8179e982acb7?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80");
background-size: 100%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
#st.sidebar.header("")

def display_sarcastic_remark(remark):
    st.title(remark)
    time.sleep(0.1)

import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the text
    tokens = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back into a clean text
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(2,2), max_features=500)

filename="trained_model.pkl"
filename1="vectorizer_model_tfidf.pkl"

with open(filename1, 'rb') as file:
   vectorizer1 = pickle.load(file)

st.header('Sentiment Analysis of a Hotel Review')
with st.title('Input Review'):
	text = st.text_input('Text here: ')
cleaned_text=clean_text(text)
input_df=pd.DataFrame([text],columns=["Review"])

def sentiment_scores(sentence):
 
    # instantiate afinn
    sid_obj = Afinn()
 
    # polarity_scores method of afinn
    sentiment_dict = sid_obj.score(sentence)
 
    # decide sentiment as positive, negative and neutral
    if sentiment_dict == 0 :
        return "Neutral review",sentiment_dict
    elif sentiment_dict > 0:
    	return "Positive review",sentiment_dict
    else :
        return "Negative review",sentiment_dict

def annotating_text(text,compound_score):
	vectorizer.fit_transform(input_df['Review'])
	annot_text=list()
	for i in vectorizer.get_feature_names_out():
		score=sentiment_scores(i)[1]
		if compound_score<0:
			if score<0:
				#st.write(i,score)
				annot_text.append((i, "Negative", "#ea9999","#010203"))
		if compound_score>0:
			if score>0:
				#st.write(i,score)
				annot_text.append((i, "Positive", "#b6d7a8","#010203"))
		if compound_score==0:
			if score==0:
				#st.write(i,score)
				annot_text.append((i, "Neutral", "#ffd966","#010203"))
	return annot_text


if text!="":
	compound_score=sentiment_scores(text)[0]
	display_sarcastic_remark(sentiment_scores(text)[0])
	#st.write(compound_score)

	# Load the model from the file
	with open(filename1, 'rb') as file:
		loaded_model = pickle.load(file)

	input1=np.stack(loaded_model.transform([cleaned_text]).toarray())




	# Load the model from the file
	with open(filename, 'rb') as file:
		prediction_model = pickle.load(file)

	Prediction_Proba=[]
	Prediction_Proba.append(str(np.round(prediction_model.predict_proba(input1)[0,prediction_model.predict(input1)[0]]*100,3))+"%")
	
	if prediction_model.predict(input1)[0]==0:
		st.write(f'Negativity Level {Prediction_Proba[0]}')
	else:
		st.write(f'Positivity Level {Prediction_Proba[0]}')

	annot_text=annotating_text(text,sentiment_scores(text)[1])
	annotated_text(annot_text)
