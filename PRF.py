from math import log, sqrt

def generateInvertedIndex(doc_dict):
    invertedIndex = {}
    tokenDict = {}
    doc_dict = doc_dict
    for line in doc_dict:
        doc_id, text = line
        doc_text = text.split()
        length = len(doc_text)
        tokenDict[doc_id] = length
        for word in text.split():
            if word not in invertedIndex.keys():
                docIDCount = {doc_id : 1}
                invertedIndex[word] = docIDCount
            elif doc_id in invertedIndex[word].keys():
                invertedIndex[word][doc_id] += 1
            else:
                docIDCount = {doc_id : 1}
                invertedIndex[word].update(docIDCount)
    return invertedIndex

def queryFrequency(query, invertedIndex):
    queryFreq = {}
    for term in query.split():
        if term in queryFreq.keys():
            queryFreq[term] += 1
        else:
            queryFreq[term] = 1
    for term in invertedIndex:
        if term not in queryFreq.keys():
            queryFreq[term] = 0
    #print(queryFreq)
    return queryFreq

def calculateDocsCount(doc, docIndex, doc_dict):
    doc_dict = doc_dict
    for line in doc_dict:
        doc_id, text = line
        if doc_id == doc:
            for term in text.split():
                if term in docIndex.keys():
                    docIndex[term] += 1
                else:
                    docIndex[term] = 1
    return docIndex

def findDocs(doc_dict, k, bm25score, invertedIndex, relevancy):
    doc_dict = doc_dict
    relIndex = {}
    nonRelIndex = {}
    if relevancy == "Relevant":
        for i in range(0, k):
            doc,doc_score = bm25score[i]
            relIndex = calculateDocsCount(doc, relIndex, doc_dict)
        for term in invertedIndex:
            if term not in relIndex.keys():
                relIndex[term] = 0
        return relIndex
    
    
    elif relevancy == "Non-Relevant":
        for i in range(k+1,len(bm25score)):
            doc,doc_score = bm25score[i]
            nonRelIndex = calculateDocsCount(doc, nonRelIndex, doc_dict)
        for term in invertedIndex:
            if term not in nonRelIndex.keys():
                nonRelIndex[term] = 0   
        return nonRelIndex
    
def findRelDocMagnitude(docIndex):
    mag = 0
    for term in docIndex:
        mag += float(docIndex[term]**2)
        mag = float(sqrt(mag))
    return mag

def findNonRelDocMagnitude(docIndex):
    mag = 0
    for term in docIndex:
        mag += float(docIndex[term]**2)
    mag = float(sqrt(mag))
    return mag

def findRocchioScore(term, queryFreq, relDocMag, relIndex, nonRelMag, nonRelIndex):
    ALPHA = 1
    BETA = 0.75
    GAMMA = 0.15
    Q1 = ALPHA * queryFreq[term] 
    Q2 = (BETA/relDocMag) * relIndex[term]
    Q3 = (GAMMA/nonRelMag) * nonRelIndex[term]
    rocchioScore = ALPHA * queryFreq[term] + (BETA/relDocMag) * relIndex[term] - (GAMMA/nonRelMag) * nonRelIndex[term]
    return rocchioScore

def findNewQuery(doc_dict, invertedIndex, query, k, bm25score, topNRocchio):
    queryFreq = queryFrequency(query, invertedIndex)
    relIndex = findDocs(doc_dict, k, bm25score, invertedIndex, "Relevant")
    relDocMag = findRelDocMagnitude(relIndex)
    nonRelIndex = findDocs(doc_dict, k, bm25score, invertedIndex, "Non-Relevant")
    nonRelMag = findNonRelDocMagnitude(nonRelIndex)
    
    updatedQuery = {}
    newQuery = query
    for term in invertedIndex:
        updatedQuery[term] = findRocchioScore(term, queryFreq, relDocMag, relIndex, nonRelMag, nonRelIndex)
    sortedUpdatedQuery = sorted(updatedQuery.items(), key=lambda x:x[1], reverse=True)
    if len(sortedUpdatedQuery) < topNRocchio:
        loopRange = len(sortedUpdatedQuery)
    else:
        loopRange = topNRocchio
    i = 0
    
    for i in range(loopRange):
        term,frequency = sortedUpdatedQuery[i]
        #print("term, frequency", term, frequency)
        if term not in query:
            newQuery +=  " "
            newQuery +=  term
            
    return newQuery