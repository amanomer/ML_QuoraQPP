<h3>Solution overview: What has been done</h3>
<h4>Load data from csv</h4>
<t>Read from csv file using pandas library</t>
<t>Analyse basic stats</t>
<ui>
  <li>Total number of data points</li>
  <li>Number of questions that similar i.e, label=1</li>
  <li>Number of unique questions</li>
  <li>Number of question pairs that are repeated</li>
  <li>Log-Histogram of question appearance counts</li>
  <li>Check null values</li>
</ui>

<h3>Basic Feature Extraction (Total=11)</h3> (Visualize and Analyse)
<ui>
  <li>freq_qid1 = Frequency of qid1's</li>
<li>freq_qid2 = Frequency of qid2's</li>
<li>q1len = Length of q1</li>
<li>q2len = Length of q2</li>
<li>q1_n_words = Number of words in Question 1</li>
<li>q2_n_words = Number of words in Question 2</li>
<li>word_Common = (Number of common unique words in Question 1 and Question 2)</li>
<li>word_Total =(Total num of words in Question 1 + Total num of words in Question 2)</li>
<li>word_share = (word_common)/(word_Total)</li>
<li>freq_q1+freq_q2 = sum total of frequency of qid1 and qid2</li>
<li>freq_q1-freq_q2 = absolute difference of frequency of qid1 and qid2</li>
</ui>
<h5>Analysis</h5>
PDF of word_share have some overlap but both curves show definite separation. Hence, word_share feature can be utilized to build model.
The average word share and Common no. of words of qid1 and qid2 is more when questions are duplicate(Similar).
The distributions of the word_Common feature in similar and non-similar questions are highly overlapping

<h3> Preprocessing of text</h3>
<ui>
  <li>Removing html tags</li>
  <li>Removing Punctuations</li>
  <li>Performing stemming (removing affixes. For example, "running," "runner," and "runs" can all be reduced to the stem "run". </li>
  <li>Removing Stopwords (like if, of...)</li>
  <li>Expanding contractions (like i'm -> i am)</li>
</ui>

<h3> Advanced Feature Extraction (NLP & Fuzzy features) (Visualize & Analyse) </h3>
<h5> Definition</h5>
Token: You get a token by splitting sentence a space <br>
Stop_Word : stop words as per NLTK. <br>
Word : A token that is not a stop_word <br>

<h5> Features (Total 15)</h5>
cwc_min : Ratio of common_word_count to min lenghth of word count of Q1 and Q2<br>
cwc_min = common_word_count / (min(len(q1_words), len(q2_words))<br>
cwc_max : Ratio of common_word_count to max lenghth of word count of Q1 and Q2<br>
cwc_max = common_word_count / (max(len(q1_words), len(q2_words))<br>
csc_min : Ratio of common_stop_count to min lenghth of stop count of Q1 and Q2<br>
csc_min = common_stop_count / (min(len(q1_stops), len(q2_stops))<br>
csc_max : Ratio of common_stop_count to max lenghth of stop count of Q1 and Q2<br>
csc_max = common_stop_count / (max(len(q1_stops), len(q2_stops))<br>
ctc_min : Ratio of common_token_count to min lenghth of token count of Q1 and Q2<br>
ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens))<br>
ctc_max : Ratio of common_token_count to max lenghth of token count of Q1 and Q2<br>
ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens))<br>
last_word_eq : Check if First word of both questions is equal or not<br>
last_word_eq = int(q1_tokens[-1] == q2_tokens[-1])<br>
first_word_eq : Check if First word of both questions is equal or not<br>
first_word_eq = int(q1_tokens[0] == q2_tokens[0])<br>
abs_len_diff : Abs. length difference<br>
abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))<br>
mean_len : Average Token Length of both Questions<br>
mean_len = (len(q1_tokens) + len(q2_tokens))/2<br>
fuzz_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/ (Value 0 to 100(Very similar))<br>
fuzz_partial_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/<br>
token_sort_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/<br>
token_set_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/<br>
longest_substr_ratio : Ratio of length longest common substring to min lenghth of token count of Q1 and Q2<br>
longest_substr_ratio = len(longest common substring) / (min(len(q1_tokens), len(q2_tokens))<br>

<h5>Analysis</h5>
Create Word Cloud of duplicate and non-duplicate question pairs to observe most frequent occuring words.<br>
Pair plot of features ('ctc_min', 'cwc_min', 'csc_min', 'token_sort_ratio') <br>
Distribution of the token_sort_ratio <br>
Distribution of the fuzz_ratio <br>

<h5> Visualise</h5>
Using TSNE for Dimentionality reduction for 15 Features(Generated after cleaning the data) to 2D & 3D <br>

<h3> Convert Word to Vector</h3>
Featurizing text data with TFIDF weighted word vector. <br>
After we find TF-IDF scores, we convert each question to a weighted average of word2vec vectors by these scores. <br>
here we use a pre-trained GLOVE model which comes free with "Spacy". https://spacy.io/usage/vectors-similarity <br>
It is trained on Wikipedia and therefore, it is stronger in terms of word semantics. <br>

<h3> Build ML Model</h3>
Use only 100K rows for quick processing.
<h5> Random Model</h5>
log loss range [0, inf). To get worst case log loss(=0.88).<br>
Build model log loss less than 0.88 (best=0).

<h5> Logistic Regression with Hyperparameter Tuning</h5>
Linear models (like LR, SVM) are used for high dimensioanlity data (=221).<br>
<code>SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)</code><br>
For values of best alpha =  1 The test log loss is: 0.520035530431 <br>
<img width="1383" height="328" alt="image" src="https://github.com/user-attachments/assets/572c1896-eaf4-4a69-bd5b-548b87a0783e" />
Precision of class1 and class2 are ok but recall of class2(=0.496) is poor.<br>

<h5> SVM with Hyperparameter Tuning</h5>
<code>SGDClassifier(alpha=i, penalty='l1', loss='hinge', random_state=42)</code> <br>
For values of best alpha =  0.0001 The test log loss is: 0.489669093534 <br>
Clearly SVM is better than LR for log loss.<br>
<img width="1421" height="330" alt="image" src="https://github.com/user-attachments/assets/994e8cca-c12d-4e8a-acbb-d143c358e7f7" />
But recall is very low(0.486).<br>
Linear model of high bias problem or underfitting problem.<br>
Therefore using complex model like DT<br>

<h5> XGBoost</h5>
<code>xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=20, verbose_eval=10)</code> <br>
The test log loss is: 0.357054433715<br>
<img width="1390" height="330" alt="image" src="https://github.com/user-attachments/assets/ce54377c-9720-4025-bcc9-4461dfb2ed7b" />


<h3> Assignment </h3>
Try out models (Logistic regression, Linear-SVM) with simple TF-IDF vectors instead of TD_IDF weighted word2Vec.<br>
Hyperparameter tune XgBoost using RandomSearch to reduce the log-loss.

