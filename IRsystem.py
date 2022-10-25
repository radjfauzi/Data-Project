import os
import csv
import numpy as np
import pandas as pd
import math
import streamlit as st

from Score import BM25
from PRF import findNewQuery, generateInvertedIndex
from Preprocessor import Preprocessor


def main():
    # Get csv
    csv_name = 'muslim-v4-prepped.csv'
    df = pd.read_csv(csv_name)
    df['hadis_number']= df['hadis_number'].apply(str)
    corpus = df.hadis_content


    # Splitting sentence into array of words
    texts = [
        [word for word in document.lower().split()]
        for document in corpus
    ]

    # Count of every word inside a dictionary
    word_count_dict = {}
    for text in texts:
        for token in text:
            word_count = word_count_dict.get(token, 0) + 1
            word_count_dict[token] = word_count

    texts = [[token for token in text if word_count_dict [token] > 1] for text in texts]


    st.title("Pencarian Hadis Shahih Muslim Bahasa Indonesia")
    st.write("Website pencarian hadis Shahih Muslim menggunakan metode BM25 dan Pseudo Relevance Feedback")


    doc_dict = df[['hadis_number','hadis_content']]
    doc_dict = doc_dict.values.tolist()
    invertedIndex = generateInvertedIndex(doc_dict)

    # build model
    model = BM25()
    model.fit(texts)

    # first retrieval
    query = st.text_input("Masukan topik yang ingin di cari:")
    processor = Preprocessor()
    prepped_query = processor.preprocess(query)

    with st.spinner("Mencari dokumen..."):
        scores = model.search(prepped_query, corpus, df)
        df_scores = pd.DataFrame(scores, columns =['score', 'number', 'prep_hadis','hadis'])

        bm25score = df_scores[['number','score']]
        bm25score = bm25score.values.tolist()
        topNRocchio = 5

        newQuery = " "

        try:
            newQuery = findNewQuery(doc_dict, invertedIndex, prepped_query, 10, bm25score, topNRocchio)
        except IndexError:
            st.write("Peringatan: Input tidak terdapat dalam korpus")


        # second retrieval
        scores = model.search(newQuery, corpus, df)
        df_scores = pd.DataFrame(scores, columns =['score', 'number', 'prep_hadis','hadis'])
        ir_result = df_scores[['number','hadis']]
        ir_result = ir_result.values.tolist()

        for i in range(len(ir_result[:10])):
            with st.expander("Shahih Muslim No. " + ir_result[i][0]):
                st.write(ir_result[i][1], end="")


if __name__ == "__main__":
    main()
