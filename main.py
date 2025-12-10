import os
import pandas as pd

#######################
###### Read data ######
#######################
df = pd.read_csv("data/raw_dataset.csv")
print("Finished reading CSV file")
df = df.dropna()

################################################
###### Basic Stats and Data Visualization ######
################################################

# from qqpsp import basic_stats
# basic_stats.print_basic_stats(df)
# basic_stats.print_unique_questions(df)

######################################
###### Basic Feature Extraction ######
######################################

if os.path.isfile('data/df_fe_without_preprocessing_train.csv'):
    dfppro = pd.read_csv("data/df_fe_without_preprocessing_train.csv",encoding='latin-1')
else:
    from qqpsp import basic_feature_extraction as basic_fe
    basic_fe.extract_features(df)

####################################
###### Analyse Basic Features ######
####################################

# from qqpsp import analyse_basic_features as abf
# abf.analyse_basic_features(df)

###########################################################
###### Preprocessing and Advanced Feature Extraction ######
###########################################################

if os.path.isfile('data/nlp_features_train.csv'):
    dfnlp = pd.read_csv("data/nlp_features_train.csv",encoding='latin-1')
else:
    from qqpsp import prep_advance_fe as pafe
    pafe.preprocess_advanced_feature_extraction(df)

#######################################
###### Analyse Advanced Features ######
#######################################

# from qqpsp import analyse_advanced_features as aaf
# aaf.analyse_af(df)

########################
###### Q_mean W2V ######
########################

if os.path.isfile('data/dataset_after_w2v.csv'):
    dfw2v = pd.read_csv("data/dataset_after_w2v.csv",encoding='latin-1')
else:
    from qqpsp import word_to_vec as w2v
    w2v.convert(df)

df1 = dfnlp.drop(['qid1','qid2','question1','question2'],axis=1)
df2 = dfppro.drop(['qid1','qid2','question1','question2','is_duplicate'],axis=1)
df3 = dfw2v.drop(['qid1','qid2','question1','question2','is_duplicate'],axis=1)
df3_q1 = pd.DataFrame(df3.q1_feats_m.values.tolist(), index= df3.index)
df3_q2 = pd.DataFrame(df3.q2_feats_m.values.tolist(), index= df3.index)

print("Number of features in nlp dataframe :", df1.shape[1])
print("Number of features in preprocessed dataframe :", df2.shape[1])
print("Number of features in question1 w2v  dataframe :", df3_q1.shape[1])
print("Number of features in question2 w2v  dataframe :", df3_q2.shape[1])
print("Number of features in final dataframe  :", df1.shape[1]+df2.shape[1]+df3_q1.shape[1]+df3_q2.shape[1])

# storing the final features to csv file
if not os.path.isfile('data/final_features.csv'):
    df3_q1['id']=df1['id']
    df3_q2['id']=df1['id']
    df1  = df1.merge(df2, on='id',how='left')
    df2  = df3_q1.merge(df3_q2, on='id',how='left')
    result  = df1.merge(df2, on='id',how='left')
    result.to_csv('data/final_features.csv')

###################################
###### Run ml_models.py file ######
###################################