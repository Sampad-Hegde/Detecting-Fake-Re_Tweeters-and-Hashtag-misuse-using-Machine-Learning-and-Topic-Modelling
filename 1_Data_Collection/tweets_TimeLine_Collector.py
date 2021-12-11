from os import listdir
from tweepy import API, OAuthHandler
from Authorizer import TwitterAuthorizer
from pandas import DataFrame


tokenFilename = 'ConsumerAuthTokens.txt'


class TweetTimeLineCollector:
    API_KEY = "<Conumer_API_KEY>"
    API_SECRETE_KEY = "<Conumer_API_SECRETE_KEY>"
    CONSUMER_API_KEY = None
    CONSUMER_API_SECRETE_KEY = None

    api = None
    userData = dict()

    tweet_headers = [
        'tweet_id',
        'auther_id',
        'user_id',
        'created_at',
        'favorite_count',
        'is_retweeted',
        'retweet_count',
        'tweet_text',
        'geo',
        'hashtags',
        'user_mentions',
        'media',
        'source',
        'language',
    ]
    user_headers = [
        'user_id',
        'name',
        'screen_name',
        'location',
        'description',
        'url',
        'is_protected',
        'followers_count',
        'friends_count',
        'created_at',
        'favourites_count',
        'verified',
        'statuses_count',
    ]

    Users_Data = []
    userList = None

    def readTokens(self):
        if tokenFilename in listdir():
            with open(tokenFilename, "r") as f:
                line = f.read().split()
                self.CONSUMER_API_KEY = line[0]
                self.CONSUMER_API_SECRETE_KEY = line[1]
        else:
            TA = TwitterAuthorizer()
            print(TA.accessTokens)
            TA.saveTokens("ConsumerAuthTokens.txt")
            self.readTokens()

    def __init__(self, userList):
        self.readTokens()
        self.userList = userList

        auth = OAuthHandler(self.API_KEY, self.API_SECRETE_KEY)
        auth.set_access_token(self.CONSUMER_API_KEY, self.CONSUMER_API_SECRETE_KEY)

        self.api = API(auth)

    def dataCollector(self):
        for id in self.userList:
            try:
                tStatus, all_tweets = self.getAllTweetsData(id)
                if tStatus:
                    self.saveTweetsData(id, all_tweets)
                    print("Data Written for id = ", id)

            except Exception as e:
                print("! Error Occured for id = ",id,"Error = ",str(e))

    def TweetsDataFrame(self, id, tweetdata):
        purified_data = {
            'tweet_id' : [],
            'auther_id':[],
            'user_id':[],
            'created_at' : [],
            'favorite_count':[],
            'is_retweeted':[],
            'retweet_count':[],
            'tweet_text':[],
            'geo':[],
            'hashtags':[],
            'user_mentions':[],
            'media':[],
            'source':[],
            'language':[],

        }
        for tweet in tweetdata:
            purified_data['tweet_id'].append(tweet.id_str)
            purified_data['auther_id'].append(tweet.author.id_str)
            purified_data['user_id'].append(tweet.user.id_str)
            purified_data['created_at'].append(tweet.created_at.strftime('%d-%m-%Y %H:%M:%S'))
            purified_data['favorite_count'].append(tweet.favorite_count)
            purified_data['is_retweeted'].append(tweet.retweeted)
            purified_data['retweet_count'].append(tweet.retweet_count)
            purified_data['tweet_text'].append(tweet.full_text.encode("utf-8").decode("utf-8"))
            if tweet.geo == None:
                purified_data['geo'].append(tweet.geo)
            else:
                purified_data['geo'].append(' '.join(str(i) for i in tweet.geo['coordinates']))

            if len(tweet.entities['hashtags']) > 0:
                purified_data['hashtags'].append(' '.join([i['text'] for i in tweet.entities['hashtags']]))
            else:
                purified_data['hashtags'].append(None)

            if len(tweet.entities['user_mentions']) > 0:
                purified_data['user_mentions'].append( ' '.join((i['id_str'] for i in tweet.entities['user_mentions'])))
            else:
                purified_data['user_mentions'].append(None)

            if 'media' in tweet.entities.keys():
                if len(tweet.entities['media']) > 0:
                    purified_data['media'].append('\n'.join([str(x) for x in [x['media_url'] for x in tweet.entities['media']]]))
                else:
                    purified_data['media'].append('None')
            else:
                purified_data['media'].append('None')
            purified_data['source'].append(tweet.source)
            purified_data['language'].append(tweet.lang)

        df = DataFrame.from_dict(purified_data)
        df.columns=self.tweet_headers
        return df

    def saveTweetsData(self,id,tweetdata):
        df = self.TweetsDataFrame(id,tweetdata)
        df.to_csv('UserWiseData//%s_tweets.csv' % id, index=False)

    def getAllTweetsData(self, id):
        all_tweets = []
        tweets = self.api.user_timeline(user_id=id, count=200, include_rts=True, tweet_mode='extended')
        if len(tweets) > 0:
            all_tweets.extend(tweets)
            oldest_id = tweets[-1].id

            while True:
                tweets = self.api.user_timeline(user_id=id, count=200, include_rts=True, max_id=oldest_id - 1, tweet_mode='extended')
                if len(tweets) == 0:
                    break
                else:
                    oldest_id = tweets[-1].id
                    all_tweets.extend(tweets)

            print('Number of tweets fetched for user id : {} = {}'.format(id,len(all_tweets)))
            return True, all_tweets
        else:
            return  False, None

    def __del__(self):
        pass
        # DataFrame(self.userData).to_csv("UsersData.csv")
