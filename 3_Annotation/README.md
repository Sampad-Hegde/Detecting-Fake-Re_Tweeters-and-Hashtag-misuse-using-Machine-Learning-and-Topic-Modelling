# Annotating the Data 

## In this project there are two types of annotation:
- Annotation of the user based on retweet activity.
- Annotation of the user based on #HashTag mis-use or tweet text is irrelevant to the topic.

### 1. Annotation of the user based on retweet activity.

see all tweet (Re-Tweet) activity of a users, and it will plot a graph you can see all points by zooming it in and out judge as you wish the user is fake re-tweeter or a genuine user.

### Running the Script

```bash
python3 annotation_driver.py
```


### 2. Annotation of the user based on #HashTag mis-use or tweet text is irrelevant to the topic.
It will print the text that is tweeted towards the topic saved inside "Users.csv" file and asks you is the tweet is genuine or fake according the topic. 


### Running the Script

```bash
python3 text_annotation.py
```