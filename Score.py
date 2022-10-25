import math

class BM25:
    """
    # Default Parameters
    - k1 : float, default 1.2
    *set higher when the text is a lot longer and more diverse

    - b : float, default 0.75
    *set lower when the text is long and causing the term less relevant
    """
	
	#change k1, b here
    def __init__(self, k1=1.2, b=0.75):
        self.b = b
        self.k1 = k1

    # Model build
    def fit(self, corpus):
        tf = []
        df = {}
        idf = {}
        doc_len = []
        corpus_size = 0

        # 4929 document
        for document in corpus:
            corpus_size += 1

            # number of chars inside a document
            doc_len.append(len(document))

            # increase term count by 1 if exists 
            # TF counter
            frequencies = {}
            for term in document:
                term_count = frequencies.get(term, 0) + 1
                frequencies[term] = term_count
            tf.append(frequencies)

            # increase term count by 1 if exists in the doc
            # put it inside doc freq count 
            for term, _ in frequencies.items():
                df_count = df.get(term, 0) + 1
                df[term] = df_count 

        # update IDF value after updating doc freq
        for term, freq in df.items():
            idf[term] = math.log(1 + (corpus_size - freq + 0.5) / (freq + 0.5))

        # assign new value of each variable
        self.tf_ = tf
        self.df_ = df
        self.idf_ = idf
        self.doc_len_ = doc_len
        self.corpus_ = corpus
        self.corpus_size_ = corpus_size
        self.avg_doc_len_ = sum(doc_len) / corpus_size
        return self

    # search method
    # TODO change the output 
    
    def search(self, query, corpus, df):
        # split query into array of terms
        prepped_query = str(query).split()
        # query = [word for word in query.lower().split()]
        # print(prepped_query)
        
        # count score for each document (4929)
        scores = [
            self._score(prepped_query, index) 
            for index in range(self.corpus_size_)
        ]

        # Add document content for each result
        results = []
        for score, doc in zip(scores, corpus):
            # round the commas into 3 number
            score = round(score, 5)
            
            if score != 0:
                # Locate the hadis_number
                df_number = df.loc[df['hadis_content'] == doc, 'hadis_number'].item()

                # print(df_number)
                
                # Locate the hadis
                prepped_hadis = df.loc[df['hadis_number'] == str(df_number), 'hadis_content'].item()

                # Locate the hadis origin
                real_hadis = df.loc[df['hadis_number'] == str(df_number), 'origin_content'].item()
                
                # print("REAL" + real_hadis)
           
                result = [score,df_number,prepped_hadis,real_hadis]

                results.append(result)
                
            results.sort(reverse=True)

        return results

    def _score(self, query, index):
        score = 0.0

        # get current doc len and current doc TF
        doc_len = self.doc_len_[index]
        frequencies = self.tf_[index]

        for term in query:
            if term not in frequencies:
                continue
            # frequency of term 't' in doc 'd'
            freq = frequencies[term]

            numerator = self.idf_[term] * freq * (self.k1 + 1)
            denominator = self.k1 * ((1 - self.b) + self.b * (doc_len / self.avg_doc_len_)) + freq
            score += (numerator / denominator)

        return score
