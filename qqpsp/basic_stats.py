import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Basic stats
def print_basic_stats(df):
    print("Number of data points:",df.shape[0])
    df.info()
    df.groupby("is_duplicate")['id'].count().plot.bar()
    print('~> Total number of question pairs for training:\n   {}'.format(len(df)))
    print('~> Question pairs are not Similar (is_duplicate = 0):\n   {}%'.format(100 - round(df['is_duplicate'].mean()*100, 2)))
    print('\n~> Question pairs are Similar (is_duplicate = 1):\n   {}%'.format(round(df['is_duplicate'].mean()*100, 2)))

#Unique Questions
def print_unique_questions(df):
    # print("Here")
    qids = pd.Series(df['qid1'].tolist() + df['qid2'].tolist())
    unique_qs = len(np.unique(qids))
    qs_morethan_onetime = np.sum(qids.value_counts() > 1)
    print ('Total number of  Unique Questions are: {}\n'.format(unique_qs))
    print ('Number of unique questions that appear more than one time: {} ({}%)\n'.format(qs_morethan_onetime,qs_morethan_onetime/unique_qs*100))
    print ('Max number of times a single question is repeated: {}\n'.format(max(qids.value_counts()))) 
    #checking whether there are any repeated pair of questions
    pair_duplicates = df[['qid1','qid2','is_duplicate']].groupby(['qid1','qid2']).count().reset_index()
    print ("Number of duplicate questions",(pair_duplicates).shape[0] - df.shape[0])

# Plot
def plot_graphs(df):
    qids = pd.Series(df['qid1'].tolist() + df['qid2'].tolist())
    unique_qs = len(np.unique(qids))
    qs_morethan_onetime = np.sum(qids.value_counts() > 1)
    # x = ["unique_questions" , "Repeated Questions"]
    # y =  [unique_qs , qs_morethan_onetime]
    # plt.figure(figsize=(10, 6))
    # plt.title ("Plot representing unique and repeated questions  ")
    # sns.barplot(x="unique_questions", y="Repeated Questions", data=pd.Dataframe(y))
    # plt.show()
    
    #Plot 2 Number of occurrences of each question
    plt.figure(figsize=(20, 10))
    plt.hist(qids.value_counts(), bins=160)
    # plt.yscale('log', nonposy='clip')
    plt.title('Log-Histogram of question appearance counts')
    plt.xlabel('Number of occurences of question')
    plt.ylabel('Number of questions')
    print ('Maximum number of times a single question is repeated: {}\n'.format(max(qids.value_counts()))) 
    plt.show()
