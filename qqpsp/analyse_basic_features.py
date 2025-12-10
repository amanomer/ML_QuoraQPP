import seaborn as sns
import matplotlib.pyplot as plt

def analyse_basic_features(df):
    print ("Minimum length of the questions in question1 : " , min(df['q1_n_words']))
    print ("Minimum length of the questions in question2 : " , min(df['q2_n_words']))
    print ("Number of Questions with minimum length [question1] :", df[df['q1_n_words']== 1].shape[0])
    print ("Number of Questions with minimum length [question2] :", df[df['q2_n_words']== 1].shape[0])
    
    # Plot word_share
    plt.figure(figsize=(12, 8))
    plt.subplot(1,2,1)
    sns.violinplot(x = 'is_duplicate', y = 'word_share', data = df[0:])
    plt.subplot(1,2,2)
    sns.distplot(df[df['is_duplicate'] == 1.0]['word_share'][0:] , label = "1", color = 'red')
    sns.distplot(df[df['is_duplicate'] == 0.0]['word_share'][0:] , label = "0" , color = 'blue' )
    plt.show()
    
    #Plot word_common
    plt.figure(figsize=(12, 8))
    plt.subplot(1,2,1)
    sns.violinplot(x = 'is_duplicate', y = 'word_Common', data = df[0:])
    plt.subplot(1,2,2)
    sns.distplot(df[df['is_duplicate'] == 1.0]['word_Common'][0:] , label = "1", color = 'red')
    sns.distplot(df[df['is_duplicate'] == 0.0]['word_Common'][0:] , label = "0" , color = 'blue' )
    plt.show()

