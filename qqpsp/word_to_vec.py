#pip install spacy
import spacy
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def convert(df):
    df['question1'] = df['question1'].apply(lambda x: str(x))
    df['question2'] = df['question2'].apply(lambda x: str(x))
    
    # merge text of questions
    questions = list(df['question1']) + list(df['question2'])
    
    tfidf = TfidfVectorizer(lowercase=False)
    tfidf.fit_transform(questions)
    
    # dict key:word and value:tf-idf score
    word2tfidf = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
    
    # After we find TF-IDF scores, we convert each question to a weighted average of word2vec vectors by these scores.
    # here we use a pre-trained GLOVE model which comes free with "Spacy". https://spacy.io/usage/vectors-similarity
    # It is trained on Wikipedia and therefore, it is stronger in terms of word semantics.
    # en_vectors_web_lg, which includes over 1 million unique vectors.
    # python -m spacy download en_core_web_sm
    nlp = spacy.load('en_core_web_sm')
    
    vecs1 = []         #2HRS_7MINS
    
    # https://github.com/noamraph/tqdm
    # tqdm is used to print the progress bar
    for qu1 in tqdm(list(df['question1'])):
    
        doc1 = nlp(qu1) #type conersion str to 'spacy.tokens.doc.Doc'
    
        # 384 is the number of dimensions of vectors 
        # Changed to 96. Think, change in dimension of vector coz of changed version of spacy
        mean_vec1 = np.zeros([len(doc1), 96])
        for word1 in doc1:
            # word2vec
            vec1 = word1.vector  # Giving 1*96 array instead of 1*384 
            # fetch df score
            try:
                idf = word2tfidf[str(word1)]
            except:
                idf = 0
            # compute final vec
            mean_vec1 += vec1 * idf
        mean_vec1 = mean_vec1.mean(axis=0)
        vecs1.append(mean_vec1)
    df['q1_feats_m'] = list(vecs1)
    
    vecs2 = []    #3HRS 14MINS
    for qu2 in tqdm(list(df['question2'])):
        if type(qu2) is float:
            qu2=''
        doc2 = nlp(qu2) 
        mean_vec2 = np.zeros([len(doc2), 96])
        for word2 in doc2:
            # word2vec
            vec2 = word2.vector
            # fetch df score
            try:
                idf = word2tfidf[str(word2)]
            except:
                idf = 0
            # compute final vec
            mean_vec2 += vec2 * idf
        mean_vec2 = mean_vec2.mean(axis=0)
        vecs2.append(mean_vec2)
    df['q2_feats_m'] = list(vecs2)
    
    df.to_csv('data/dataset_after_w2v.csv')