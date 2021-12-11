import os
import pandas as pd
import warnings

warnings.simplefilter(action='ignore')


def saveFile(df, fileName):
    df.to_csv(fileName)


data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '\\0_DataSet'
user_file = data_path + '\\DataSetAnnotedUsers.csv'

user_df = pd.read_csv(user_file)
# print(user_df.shape)

for i in range(user_df.shape[0]):
    if user_df.iloc[i]['Tag'] != 1:
        # print(user_df.iloc[i]['Tag'])
        print("\n----------------------------------------------------------------------------------------------------")
        print('Iteration Number : ', i)
        print(user_df.iloc[i]['tweet_text'])
        print('\n')
        while (1):
            ip = int(input('Is this tweet is irrelevant to the topic? (0 : relevant / 1: irrelevant) :'))
            if ip == 0 or ip == 1:
                break
            else:
                print('!!! Invalid Input !!!')
        user_df.iloc[i]['Tag'] = ip
        print("----------------------------------------------------------------------------------------------------\n")

saveFile(user_df,data_path+"FinalAnnotatedUsers.csv")
