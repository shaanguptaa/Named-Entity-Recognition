from tensorflow.keras import models
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import numpy as np
import streamlit as st
from annotated_text import annotated_text

# page config
st.set_page_config(page_title=" Named Entity Recognition - Sahil Gupta", page_icon="🚀", layout="centered")

# load model
@st.cache_resource
def load_model():
    try:
        model = models.load_model('model_cpu')
        # model._make_predict_function()
        # print('model loaded')
        return model
    except Exception as e:
        raise Exception('Error loading ML Model')


# load data (word2idx, etc)
@st.cache_resource
def load_data():
    try:
        (word2idx, max_len, tags) = np.load('data.npy', allow_pickle=True)
        # print('data loaded')
        return word2idx, max_len, tags
    except Exception as e:
        raise Exception('Error loading Resources')

# styles
st.markdown("""
        <style>
            .main {
                overflow: hidden;
            }
            button {
                position: relative;
                float: right;
                transition: all 0.15s ease;
            }
            button:hover {
                border-color: green !important;
                background-color: green !important;
                color: white !important;
            }
            button[data-testid=StyledFullScreenButton] {
                display: none !important;
            }
            span {
                margin: 0.3em 0.3em 0 0.1em !important;
                line-height: 1.3em !important;
            }
            .footer > p > a {
                color: white !important;
                text-decoration: none !important;
                display: inline-block;
                font-weight: bold;
                transition: all 0.2s ease-out;
            }
            .footer > p > a::after {
                content: "";
                height: 1px;
                width: 0;
                display: block;
                margin: auto;
                background-color: white;
                transition: all 0.2s ease-out;
            }
            .footer > p > a:hover::after {
                width: 100%;
            }
            .footer > p > a > img {
                margin: auto 0.3em !important;
            }
        </style>
                
            """, unsafe_allow_html=True)

# title and caption
st.markdown('<p style="margin-top: -0.5em; text-align: center; color: white; font-size: 3.5em; font-weight: 600;">Named Entity Recognition</p>', unsafe_allow_html=True)

# input area
text_area = st.text_area(label='Input Sentence', placeholder="Enter Sentence", value='', label_visibility="hidden")

col1, col2 = st.columns((0.8, 0.2))
col1.empty()
col2.empty()

# button
btn = col2.button(label='Submit', type='secondary')

# spacing
st.container().empty()

# output container
output_container = st.container().empty()


# preprocess text
def preprocess_text(input_text, word2idx, max_len):
    tokens = nltk.word_tokenize(input_text)
    # print('data preprocessed')
    return pad_sequences(sequences=[[word2idx.get(w, 0) for w in tokens]], padding="post", value=0, maxlen=max_len), tokens

# Post Process (grouping words e.g. B-gpe and next immediate I-gpe will be combined as one B-gpe)
def group_words(words):
    temp = [words[0]]
    for item in words[1:]:
        if 'I-' in item[1]:
            temp[-1][0] += ' ' + item[0]
        else:
            temp.append(item)
    
    # converting tags-names to words (B-per -> Person, etc)
    group = []
    for word, tag in temp:
        word = word + ' '
        if tag == 'O':
            group.append(word)
        elif tag == 'B-nat':
            group.append((word, 'Nationality'))
        elif tag == 'B-org':
            group.append((word, 'Organization'))
        elif tag == 'B-art':
            group.append((word, 'Art'))
        elif tag == 'B-tim':
            group.append((word, 'Time'))
        elif tag == 'B-geo':
            group.append((word, 'Location'))
        elif tag == 'B-eve':
            group.append((word, 'Event'))
        elif tag == 'B-gpe':
            group.append((word, 'Geo-Political'))
        elif tag == 'B-per':
            group.append((word, 'Person'))

    # print('data grouped')
    return group

# post process
def postprocess(prediction, tags, processed_input):
    prediction = np.argmax(prediction, axis=-1)
    prediction = [[w, tags[p]] for w, p in zip(processed_input, prediction[0])]
    groups = group_words(prediction)

    # remove 'O' tags and make in format for annotated_text
    groups = [item[0] if item[1] == 'O' else item for item in groups]
    
    # print('data postprocessed')
    return groups

# predict
def predict(input_text):
    try:
        # load model
        model = load_model()
        # load data
        word2idx, max_len, tags = load_data()
        # preprocess text
        ids, tokens = preprocess_text(input_text, word2idx, max_len)
        # predict
        prediction = model.predict(np.array([ids[0]]))
        # print('data predicted')
        # postprocess
        output = postprocess(prediction, tags, tokens)
        
        return output
    except Exception as e:
        raise e


# handle submit event
def handle_submit():
    with col1, st.spinner(''):
        # col1.image("spinner.gif", width=24)
        input_text = text_area.strip()
        try:
            if input_text == '':
                raise Exception('Please enter a sentence')            
            result = predict(input_text)
            with output_container:
                annotated_text(*result)
        except Exception as e:
            output_container.error(e.__str__())

# handle button click
if btn:
    handle_submit()

# footer
st.markdown("""
    <div class="footer" style="position: fixed; bottom: 0; left: 0; width:100%; height:fit-content; z-index: 1000;">
        <p style="width: 100%; text-align: center; height:fit-content;">
            Designed & Developed by <a href="" target="_blank">Sahil Gupta</a> &nbsp; | &nbsp; 
            <a href="https://www.linkedin.com/in/sahilguptaa/" target="_blank"> <img src="app/static/linkedin.png" alt="LinkedIn" width=20> </a>
            <a href="https://github.com/shaanguptaa" target="_blank"> <img src="app/static/github.png" alt="Github" width=20> </a>
            <a href="https://www.instagram.com/shaan_gupta_/" target="_blank"> <img src="app/static/instagram.png" alt="Instagram" width=20> </a>
        </p>
    </div>
""", unsafe_allow_html=True)

