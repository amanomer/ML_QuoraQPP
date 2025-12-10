<h1>Quora question pair problem</h1>

Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.

Credits: Kaggle

<h3>Problem Statement</h3>
<ul>
  <li>Identify which questions asked on Quora are duplicates of questions that have already been asked.</li>
  <li>This could be useful to instantly provide answers to questions that have already been answered.</li>
  <li>We are tasked with predicting whether a pair of questions are duplicates or not.</li>
</ul>
 

Source: https://www.kaggle.com/c/quora-question-pairs

<h3>Real world/Business Objectives and Constraints</h3>
<ul>
  <li>The cost of a mis-classification can be very high.</li>
  <li>We want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.</li>
  <li>No strict latency concerns.</li>
</ul>

<h3>Data Overview</h3>
<ul>
  <li>Data will be in a file Train.csv</li>
  <li>Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate</li>
  <li>Size of Train.csv - 60MB</li>
  <li>Number of rows in Train.csv = 404,290</li>
</ul>

<h3>Performance Metric</h3>
<ul>
  <li>Log loss</li>
  <li>Binary Confusion Matrix</li>
</ul>

<h3>Environment Details</h3>
Python 

Windows


<h3>How to do it again?</h3>
Execute <code>main.py</code> <br>
Check whether 'final_feature.csv' is present inside data folder<br>
Now, execute <code>ml_model.py</code>

<h3>What have I learned?</h3>

