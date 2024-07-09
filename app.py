import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model and vectorizer
model = joblib.load('lr_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize PorterStemmer and stopwords
stemmer = PorterStemmer()
stopwords = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()  # Convert text to lowercase
    text = re.sub(r'\[.*?\]', '', text)  # Remove square brackets and content within them
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove newline characters
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    text = ' '.join([word for word in text.split() if word not in stopwords])  # Remove stopwords
    text = ' '.join([stemmer.stem(word) for word in text.split()])  # Apply stemming
    return text

# Set the title and description of the app
st.set_page_config(page_title='Fake News Classifier', page_icon='📰')
st.title('📰 Fake News Classifier')
st.write('Enter the news text below to classify it as Real or Fake.')

# Add a sidebar for user input
st.sidebar.header('Input Options')
st.sidebar.write('You can paste the news text into the input box below.')

# Add an expander for instructions
with st.expander("See Instructions"):
    st.write("""
        1. Enter the news text in the text box.
        2. Click the 'Classify' button to get the prediction.
        3. The result will display whether the news is Real or Fake.
    """)

# Text input
user_input = st.text_area('News Text', height=200)

if st.button('Classify'):
    if user_input.strip() == "":
        st.warning('Please enter some text to classify.')
    else:
        # Clean the input text
        cleaned_input = clean(user_input)
        
        # Transform the input text using the TF-IDF vectorizer
        input_vector = vectorizer.transform([cleaned_input])
        
        # Make a prediction
        prediction = model.predict(input_vector)[0]
        
        # Display the prediction with color coding
        if prediction == 1:
            st.success('The news is **Real**.')
        else:
            st.error('The news is **Fake**.')