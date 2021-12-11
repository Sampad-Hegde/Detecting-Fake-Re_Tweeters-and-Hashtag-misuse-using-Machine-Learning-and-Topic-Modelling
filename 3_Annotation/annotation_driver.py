import os
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.simplefilter(action='ignore')
plt.style.use('fivethirtyeight')


def dateTimeCreator(s):
    s = s.split(' ')
    year, month, day = s[0].split('-')
    hour, minu, seco = s[1].split(':')
    return datetime(int(year), int(month), int(day), int(hour), int(minu), int(seco))


def hourReturner(totSec):
    return (totSec / 24) / 60


def normalize(df):
    """Normalize the DF using min/max"""
    Scaler = MinMaxScaler(feature_range=(0, 1))
    dates_scaled = Scaler.fit_transform(df.created_at.values.reshape(-1, 1)).reshape(1, -1)[0]
    # print(dates_scaled)
    return dates_scaled


def time_delta_calc(t1, t0):
    return t0 - t1


def saveData(ulist, tlist):
    pd.DataFrame(list(zip(ulist, tlist)), columns=['UserID', 'TAG']).to_csv("AnnotedData.csv")


def writeAnnotation(users, tags):
    users_df['Tag'] = np.nan
    for i in range(len(users)):
        # idx = users_df.user_id.where(users_df.user_id == users[i])
        idx = users_df.index[users_df['user_id'] == users[i]].tolist()
        if len(idx) > 1:
            print("More than 1 User ID Macthed")
        else:
            idx = idx[0]

            users_df.Tag.iloc[idx] = tags[i]
            print("Written Tag = ", users_df.iloc[idx].Tag, "For User ID = ", users[i])
    print("\n\n\n ================================================================================================\n\n")
    print("!!!! Done Annotation !!!!")
    print("\n\n\n ================================================================================================\n\n")
    users_df.to_csv(data_path + 'AnnotaedUsers.csv')


data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '\\0_DataSet'
user_file = data_path + '\\Users.csv'
userWiseDataFolder = os.path.normpath(data_path + '\\UserWiseData')
userWiseDataFiles = next(os.walk(userWiseDataFolder), (None, None, []))[2]

users_df = pd.read_csv(user_file)

tagList = []
useridList = []

ctr = 1

for file in userWiseDataFiles:
    file_path = userWiseDataFolder + '\\' + file
    tweet_df = pd.read_csv(file_path)

    tweet_df.drop(tweet_df[tweet_df['tweet_text'].str[:2] == 'RT'].index, inplace=False)

    tweet_df.created_at = tweet_df.created_at.apply(dateTimeCreator)
    start_time = tweet_df['created_at'][0]
    tweet_df = tweet_df.sort_values(by='created_at')

    # print(tweet_df['created_at'][0] - max_time)
    tweet_df.created_at = tweet_df.created_at.apply(time_delta_calc, t0=start_time)

    total_hours = (tweet_df.created_at[tweet_df.created_at.shape[0] - 1].days + 2) * 24

    bins = [pd.Timedelta(hours=x) for x in range(total_hours)]

    counts = pd.cut(tweet_df.created_at, bins).value_counts()
    # print(bins.shape)

    # print(counts.tolist())
    bin = np.linspace(-5, 5, 25, endpoint=True)
    fig, ax = plt.subplots()
    x = [x for x in range(total_hours-1)]
    # sns.countplot(x= bins, y = counts.tolist())
    plt.bar(x,counts.tolist())
    plt.show()

    print("\n-------------------------------------------------------------------------")
    print("Counter = ", ctr)
    print("Maximum tweets in per time intervals = ", max(counts))
    while(1):
        tagoftheuser = int(input("Is the user is real or fake (0 for real /1 for fake) : "))
        if(tagoftheuser == 1 or tagoftheuser ==0):
            break
        else:
            print(" !!!!   Invalid Input Enter Again   !!!! ")
    print("---------------------------------------------------------------------------\n\n")
    useridList.append(int(file.split('_')[0]))
    tagList.append(tagoftheuser)
    ctr += 1

saveData(useridList, tagList)
writeAnnotation(useridList, tagList)
