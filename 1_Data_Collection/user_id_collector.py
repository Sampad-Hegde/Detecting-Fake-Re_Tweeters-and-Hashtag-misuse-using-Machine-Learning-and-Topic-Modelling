from os import listdir
import pandas as pd
from tweepy import API, OAuthHandler, Cursor
from Authorizer import TwitterAuthorizer
from io import open

tokenFilename = 'ConsumerAuthTokens.txt'


class UserIDCollector:
    API_KEY = "<Conumer_API_KEY>"
    API_SECRETE_KEY = "<Conumer_API_SECRETE_KEY>"
    CONSUMER_API_KEY = None
    CONSUMER_API_SECRETE_KEY = None
    userIDSet = set()
    columns = ['tweet_id', 'tweet_text', 'user_id',
               'name', 'screen_name', 'location',
               'description', 'url', 'is_protected',
               'followers_count', 'friends_count', 'created_at',
               'favourites_count', 'verified', 'statuses_count']
    user_df = pd.DataFrame(columns=columns)

    def __init__(self, query):
        self.query = query
        auth = OAuthHandler(self.API_KEY, self.API_SECRETE_KEY)
        auth.set_access_token(self.CONSUMER_API_KEY, self.CONSUMER_API_SECRETE_KEY)
        self.api = API(auth)

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

    def getUserIds(self, number_pages):
        # ,  per_page = 100, result_type="recent", lang="en-us", tweet_mode = "extended"
        for page in Cursor(self.api.search, q=self.query, result_type="recent", lang="en-us",
                           tweet_mode="extended").pages(number_pages):
            for d in page:
                purified_user = self.getUserData(d)
                if purified_user != 0:
                    self.appendUserData(purified_user)
                    self.userIDSet.add(purified_user['user_id'])
                    print(f"Data is Written for id = {purified_user['user_id']}")

    def appendUserData(self, data):
        self.user_df.loc[len(self.user_df.index)] = data.values()

    def saveUserData(self, fileName):
        self.user_df.to_csv(fileName)

    def getUserData(self, uData):
        # uData = self.api.get_user(user_id=id)

        purified_user = dict()

        purified_user['tweet_id'] = uData.id
        purified_user['tweet_text'] = (uData.full_text.encode("utf-8").decode("utf-8")).replace('\n', ' ')

        uData = uData.author

        if uData.id in self.userIDSet:
            return 0
        self.userIDSet.add(uData.id)
        purified_user['user_id'] = (uData.id_str)
        purified_user['name'] = (uData.name)
        purified_user['screen_name'] = (uData.screen_name)

        if uData.location == '':
            purified_user['location'] = 'None'
        else:
            purified_user['location'] = uData.location.replace(',', '')

        if uData.description == '':
            purified_user['description'] = 'None'
        else:
            purified_user['description'] = uData.description.replace('\n', ' ')

        if uData.url == '':
            purified_user['url'] = 'None'
        else:
            purified_user['url'] = (uData.url)

        purified_user['is_protected'] = (uData.protected)
        purified_user['followers_count'] = (uData.followers_count)
        purified_user['friends_count'] = (uData.friends_count)

        purified_user['created_at'] = (uData.created_at).strftime('%d-%m-%Y %H:%M:%S')
        purified_user['favourites_count'] = (uData.favourites_count)
        purified_user['verified'] = (uData.verified)
        purified_user['statuses_count'] = (uData.statuses_count)
        return purified_user

    def getUserIdList(self):
        return list(self.userIDSet)
