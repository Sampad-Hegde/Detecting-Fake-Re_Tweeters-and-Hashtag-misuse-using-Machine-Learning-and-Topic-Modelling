# File 'Authorizer.py' generated in pycharm v2021.1 by sampad at 5:05pm 16/05/2021

from tweepy import API, OAuthHandler
import webbrowser


class TwitterAuthorizer:
    CONSUMER_API_KEY = '4jgj5TWqrQUIvVEAitGUPuiWC'
    CONSUMER_API_SECRETE_KEY = 'RfWNwy8zAELkBiTJVEfcM3OaNhybHjSELNPZGLM0l4lC0PTheQ'
    callback_URI = "oob"
    accessTokens = None

    def __init__(self):
        auth = OAuthHandler(self.CONSUMER_API_KEY, self.CONSUMER_API_SECRETE_KEY, self.callback_URI)
        redirect_url = auth.get_authorization_url()
        webbrowser.open(redirect_url)
        oneTimeAuthCode = input("Enter the Token Code : ")
        self.accessTokens = auth.get_access_token(oneTimeAuthCode)
        self.api = API(auth)

    def saveTokens(self,filename):
        if self.accessTokens != None:
            with open(filename,"w") as f:
                f.write(self.accessTokens[0])
                f.write('\n')
                f.write(self.accessTokens[1])
                print(filename,"File Written with access tokens successfully ")
        else:
            print("Error while generating the tokens :(")

