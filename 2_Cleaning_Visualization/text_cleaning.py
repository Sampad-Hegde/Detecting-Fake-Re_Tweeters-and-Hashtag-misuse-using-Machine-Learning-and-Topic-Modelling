from re import sub as substitute, MULTILINE,compile
from nltk.corpus import stopwords
import nltk
import warnings

warnings.simplefilter(action='ignore')
# nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

url = compile(r"https?:*/+[a-zA-Z0-9./]*")


def text_clean(tweet_text):
    tweet_text = substitute(r'@[A-Za-z0-9]+', '', tweet_text)
    tweet_text = substitute(r'@[A-Za-zA-Z0-9]+', '', tweet_text)
    tweet_text = substitute(r'@[A-Za-z]+', '', tweet_text)
    tweet_text = substitute(r'@[-)]+', '', tweet_text)
    tweet_text = substitute(r'#', '', tweet_text)

    tweet_text = substitute(r'RT[\s]+', '', tweet_text)
    tweet_text = substitute(r'https?\/\/\S+', '', tweet_text)
    tweet_text = substitute(r'&[a-z;]+', '', tweet_text)
    tweet_text = substitute(r'[^\w\s]', '', tweet_text)
    tweet_text = substitute(r'^https?:\/\/.*[\r\n]*', '', tweet_text, flags=MULTILINE)
    tweet_text = substitute(r'[0-9]', '', tweet_text)

    tweet_text = tweet_text.replace(': ', '')
    tweet_text = tweet_text.replace('_', '')
    tweet_text = tweet_text.replace('\n', '')
    tweet_text = tweet_text.replace('  ', ' ')
    tweet_text = tweet_text.lower()

    tweet_text = tweet_text.split()
    tweet_text = [word for word in tweet_text if word not in stop_words]
    tweet_text = " ".join(tweet_text)

    return tweet_text


def hashtagCounter(h):
    if h == 'None':
        return 0
    else:
        h = h.split(' ')
        return len(h)
