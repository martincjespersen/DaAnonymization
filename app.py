import pandas as pd
import numpy as np
import streamlit as st

from textprivacy import TextAnonymizer, TextPseudonymizer

st.set_page_config(layout="wide")

st.sidebar.image("docs/imgs/header.png", use_column_width=True)
st.sidebar.write("Anonymizing danish text using DaCy.")


TRANSFORM_TYPE = st.sidebar.selectbox(
    'Masking technique',
    ('Anonymization', 'Pseudonymization')
)

cols = ["Person", "Location", "Organization", "Miscellaneous", "CPR", "Telephone Numbers", "Emails", "Numbers"]
default_entities = ["Person", "Location", "Organization", "CPR", "Telephone Numbers", "Emails"]
ENTITIES_TO_MASK = st.sidebar.multiselect("Entities to mask", cols, default=default_entities)

NOISY_NUMBERS = False
EPSILON = None
if 'Numbers' in ENTITIES_TO_MASK:
    NOISY_NUMBERS = st.sidebar.checkbox('Apply noise to numbers instead of masking')
if NOISY_NUMBERS:
    EPSILON = st.sidebar.slider('Epsilon value (low = more noise)', min_value=0.001, max_value=10., value=None)

st.title("Remove personal information from text")
st.markdown("This allows for easy showcase of the DaAnonymization package to anonymize Danish text with relative robustness towards other languagues.")

INPUT = st.text_area('Text to mask:', "")
masked_corpus = ['']
if st.button('Mask text'):
    MASK_NUMS = True if 'Numbers' in ENTITIES_TO_MASK else False
    MASK_MISC = True if 'Miscellaneous' in ENTITIES_TO_MASK else False
    if TRANSFORM_TYPE == 'Anonymization':      
        mask_transformer = TextAnonymizer([INPUT], mask_misc=MASK_MISC, mask_numbers=MASK_NUMS, epsilon=EPSILON)
    else:
        mask_transformer = TextPseudonymizer([INPUT], mask_misc=MASK_MISC, mask_numbers=MASK_NUMS, epsilon=EPSILON)

    MASKING_ORDER =["CPR", "TELEFON", "EMAIL", "NER"]
    masked_corpus = mask_transformer.mask_corpus(masking_order=MASKING_ORDER)

st.text_area('Output:', masked_corpus[-1])