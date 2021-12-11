# Building the model and training and testing the model.

### Files which shows the results All other *.py files are for helper files or files for running the server : 
1. Temporal_Features_Models.ipynb
2. Textual_Features_Models.ipynb
3. Cobined_Models.ipynb

### ** All Trained models are available inside the ```saved_models``` folder

# Change the appropriate file path in these files.
1. Hawkes_Process.py
2. LDA.py
3. Temporal_models.py
4. Topic_Modelling.py
5. Combined_Models.py

### There Two types features are involved
- Temporal Features : 
  - This is extracted from Re-Tweet Time-Line (Time Dependent Data)
  - Re-tweet data is from "<User-ID>_tweets.csv" file because we need all his tweet activity data.
  - example : 
    - a user is re-tweeted at these time-stamps 
    - [10:30, 10:45, 10:46, 10:52, 11:01]
    - it is converted to minute passed from 1 st retweet:
    - [0,15,16,22,31]
    - apply Min-Max (0, 1) Scaler 
    - [0,0.2.0.24,0.52,0.8,1] <-- is the temporal feature
    
- Textual Features
  - This is extracted from textual information in the tweet.
  - tweet is selected from "Users.csv" because tweet should be from same topic.
  - these tweets text are already cleaned
  - example :
    - Tweet-text before cleaning : "this #covid pandemic was change my life. @mention_me"
    - Tweet-text after cleaning : "covid pandemic change my life"
    - After Tokenization : ['covid', 'pandemic', 'change', 'my', 'life']
    - for LDA we directly feed this dataset of tokenized tweets.
    - for Bag of words and Word-Embeddings:
      - create a vocabulary of unique words in the dataset.
      - replace the words with its indices.
      - ['covid', 'pandemic', 'change', 'my', 'life'] = [12, 10254, 125, 365, 854]

## Detailed Results of all model like Confusion matrix, Accuracy, Precision, Recall, F1-Score on both Train Set and Test Set is in the Results Folder


## Models only on temporal features :
First we feed the temporal features to hawkes process (Hawkes Expression Kernel)
![]("hawkes_process.png")

Output : Intensity function (BaseLine) and adjacency value 
example: <br>
input = [0,0.2.0.24,0.52,0.8,1,....] <br>
Hawkes process output = [8.45,0.245] <- baseline, Adj_value
output = fake or genuine based on temporal feature

Classifiers :

| Sl. No. | Model | Training Accuracy | Validation Accuracy |
| :--- | :----: | :----: | ---: |
| 1 | KNN | 86.6% | 88.5% |
| 2 | SVM | 86.81% | 89.21% |
| 3 | Naive Bayes | 86.81% | 89.21% |
| 4 | Fully Connected Neural Network | 86.81% | 89.54% |

## Models only on textual features :

### LDA Technique
Two dictionaries and two LDA models are created. one for Fake texts and one for Genuine texts. 
Output : Topic vectors 10 topics from genuine LDA model and 10 topics from fake LDA Model
example: <br>
input = ['covid', 'pandemic', 'change', 'my', 'life']<br>
LDA output = Genuine topics vector => [0.021, 0.01, 0.14, 0.013, 0.112, 0.5, 0.01, 0.0254,0.014, 0.155] <- probability that given text belongs to ith topic <br>
           fake topics Vectors => [0.5, 0.01, 0.0254,0.014, 0.155,0.021, 0.01, 0.14, 0.013, 0.112 ] <- probability that given text belongs to ith topic <br>
        output vector = [genuine topics vector + fake topics vector] = [0.021, 0.01, 0.14, 0.013, 0.112, 0.5, 0.01, 0.0254, 0.014, 0.155, 0.5, 0.01, 0.0254,0.014, 0.155,0.021, 0.01, 0.14, 0.013, 0.112] <br>
output : Fake or Genuine in the context of Textual Features

Classifiers :

| Sl. No. | Model | Training Accuracy | Validation Accuracy |
| :--- | :----: | :----: | ---: |
| 1 | KNN | 90.82% | 88.88% |
| 2 | SVM | 85% | 88.21% |
| 3 | Naive Bayes | 79% | 83.21% |
| 4 | Fully Connected Neural Network  | 91% | 91.83% |

### Bag Words Technique : 
For each tweet-text for list of words represented as corresponding vocab index of the word.
example: <br>
input = [12, 10254, 125, 365, 854] <br> (padded Sequence to make all tweet length to be same)
output = [0.235,0.775] <- Probabilities that belongs to the class (fake, genuine)


Classifiers :

| Sl. No. | Model | Training Accuracy | Validation Accuracy |
| :--- | :----: | :----: | ---: |
| 1 | LSTM Layers | 99.03% | 88.76% |
| 2 | Bi-LSTM, Embedding, Dense, Flatten, Dropout and Batch Normalization | 92.39% | 85.5% |

### Words Embedding Technique (Word2Vec):
For each tweet-text for list of words represented as corresponding vocab index of the word, and it is converted to pad Sequence.
in the embedding layer we give the embedding matrix as weights and make that layer non-trainable.

example: <br>
input = [12, 10254, 125, 365, 854]   (padded Sequence to make all tweet length to be same) <br>
output = [0.235,0.775] <- Probabilities that belongs to the class (fake, genuine)


Classifiers :

| Sl. No. | Model | Training Accuracy | Validation Accuracy |
| :--- | :----: | :----: | ---: |
| 1 | Bi-LSTM, Embedding (with w2v weights), Dense, Flatten, Dropout and Batch Normalization | 84.62% | 82.92% |


## Models on temporal features + Textual Features (Combined Features) :
First we feed the temporal features to hawkes process (Hawkes Expression Kernel)
example: <br>
input = LDA Topic Vectors (Size = 20 (Genuine + Fake)) + Hawkes Output (Size = 2) <br>
output =  Fake or Genuine

Classifiers :

| Sl. No. | Model | Training Accuracy | Validation Accuracy |
| :--- | :----: | :----: | ---: |
| 1 | KNN | 82.06% | 76.51% |
| 2 | SVM | 75.34% | 80.71% |
| 3 | Naive Bayes | 70.51% | 69.93% |
| 4 | Fully Connected Neural Network | 82.69% | 75.16% |
