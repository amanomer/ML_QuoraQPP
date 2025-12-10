import numpy as np
from os import path
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

def analyse_af(df):
    #Plotting Word Cloud
    df_duplicate = df[df['is_duplicate'] == 1]
    dfp_nonduplicate = df[df['is_duplicate'] == 0]
    
    # Converting 2d array of q1 and q2 and flatten the array: like {{1,2},{3,4}} to {1,2,3,4}
    p = np.dstack([df_duplicate["question1"], df_duplicate["question2"]]).flatten()
    n = np.dstack([dfp_nonduplicate["question1"], dfp_nonduplicate["question2"]]).flatten()
    
    #Saving the np array into a text file
    np.savetxt('data/train_p.txt', p, delimiter=' ', fmt='%s', encoding='utf-8')
    np.savetxt('data/train_n.txt', n, delimiter=' ', fmt='%s', encoding='utf-8')
    
    # reading the text files and removing the Stop Words:
    d = path.dirname('.')
    textp_w = open(path.join(d, 'data/train_p.txt'), encoding='utf-8').read()
    textn_w = open(path.join(d, 'data/train_n.txt'), encoding='utf-8').read()
    stopwords = set(STOPWORDS)
    stopwords.add("said")
    stopwords.add("br")
    stopwords.add(" ")
    
    stopwords.remove("not")
    stopwords.remove("no")
    stopwords.remove("like")
    
    wc = WordCloud(background_color="white", max_words=len(textp_w), stopwords=stopwords)
    wc.generate(textp_w)
    print ("Word Cloud for Duplicate Question pairs")
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    
    wc = WordCloud(background_color="white", max_words=len(textn_w), stopwords=stopwords)
    wc.generate(textn_w)
    print ("Word Cloud for Non-Duplicate Question pairs")
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    
    # Pair plot of features ['ctc_min', 'cwc_min', 'csc_min', 'token_sort_ratio'] #875sec
    n = df.shape[0]
    sns.pairplot(df[['ctc_min', 'cwc_min', 'csc_min', 'token_sort_ratio', 'is_duplicate']][0:n], hue='is_duplicate', vars=['ctc_min', 'cwc_min', 'csc_min', 'token_sort_ratio'])
    plt.show()
    
    # Distribution of the token_sort_ratio
    plt.figure(figsize=(10, 8))
    plt.subplot(1,2,1)
    sns.violinplot(x = 'is_duplicate', y = 'token_sort_ratio', data = df[0:] , )
    plt.subplot(1,2,2)
    sns.distplot(df[df['is_duplicate'] == 1.0]['token_sort_ratio'][0:] , label = "1", color = 'red')
    sns.distplot(df[df['is_duplicate'] == 0.0]['token_sort_ratio'][0:] , label = "0" , color = 'blue' )
    plt.show()
    
    # Distribution of the fuzz_ratio
    plt.figure(figsize=(10, 8))
    plt.subplot(1,2,1)
    sns.violinplot(x = 'is_duplicate', y = 'fuzz_ratio', data = df[0:] , )
    plt.subplot(1,2,2)
    sns.distplot(df[df['is_duplicate'] == 1.0]['fuzz_ratio'][0:] , label = "1", color = 'red')
    sns.distplot(df[df['is_duplicate'] == 0.0]['fuzz_ratio'][0:] , label = "0" , color = 'blue' )
    plt.show()
    
