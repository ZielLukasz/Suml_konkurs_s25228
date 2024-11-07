import streamlit as st
import pandas as pd
import time
import matplotlib as plt
import os

import streamlit as st
from transformers import pipeline

option = st.selectbox(
    "Opcje",
    [
        "Angielski -> Niemiecki",
        "???",
    ],
)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

def ButtonToGerman(text):
    
    if text:
        inf = st.info("Loading...")
        png = st.image("thinking.png",width=100)
        time.sleep(1)
        if len(text)>512:
            st.error("Za dużo znaków, ogranicz się do 512!")
            st.image("angry.png",width=100)
            inf.empty()
            png.empty()
        else:
            input_ids = tokenizer("translate English to German: "+text, return_tensors="pt").input_ids        
            outputs = model.generate(input_ids)
            st.write(tokenizer.decode(outputs[0], skip_special_tokens=True))
            inf.empty()
            st.success("Success!!! -> Erfolg!!!")
            st.image("smiley.png",width=100)
            png.empty()


st.title('Tłumacz \nLaboratorium Nr 5')
st.subheader('Napisz po angielsku w polu i naciśnij przycisk, jeżeli będzie powyżej 512 znaków wyskoczy błąd')
if option == "Angielski -> Niemiecki":
    text = st.text_area(label="Wpisz tekst")
    st.button("Click Me!", on_click=ButtonToGerman(text))
st.info("S25228")
        
