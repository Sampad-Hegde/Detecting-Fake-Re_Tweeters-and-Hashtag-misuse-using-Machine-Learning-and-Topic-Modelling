## 1 Data Collection :
All the data is directly collected from twitter itself by creating a Twitter developer account.
our topic of interest was "Covid-19 or Corona" very popular in year 2020-2021. So, We collected the data by
using Twitter API. <br><br>
Step 1:
- Collect the tweets who have used the hashtag #CORONA, #COVID-19, #CORONA-VIRUS etc in their tweets.
- select the unique users from the above list and save it into "**Users.csv"**
- Extracted Features :
    - tweet id
    - tweet text
    - user id
    - name
    - screen name (username / profile name)
    - location
    - description (like bio about user)
    - url (url to user profile)
    - is protected
    - follower count
    - friends count
    - created at (user account creation time)
    - favourites count (likes)
    - is verified profile
    - statues count (tweet/re-tweet counts)


step 2 :
- for each user in "**Users.csv"** collect all his/her recent tweets/ retweets and save that in to **"User-Wise-Data"** Folder and the name as **"{user-id}_tweets.csv"** file
- Tweet Data of a tweet contains the following Features:
  - tweet_id
  - auther_id
  - user_id
  - created_at
  - favorite_count
  - is_retweeted
  - retweet_count
  - tweet_text
  - geo
  - hashtags
  - user_mentions
  - media
  - source
  - language

step 3 :
- move ***"Users.csv"*** and ***"userWiseData"*** folder into a new folder ***"0_Dataset"*** along with the ***"1_Data_Collection"*** and ***"2_Cleaning_Visualizaion"*** etc 
## Running the Data Collection Program
place your Twitter API Credentials into ***"user_id_collector.py"*** and ***"tweets_TimeLine_Collector.py"*** files

```python3
API_KEY = "<YOUR API KEY>"
API_SECRETE_KEY = "<YOUR API SECRETE KEY>"
```
### Running the Script : 

```bash
python3 Main_Driver.py
```

#### !! This process takes lot of time nearly 24 Hours Be Careful it should not be interrupted