from tweets_TimeLine_Collector import TweetTimeLineCollector
from user_id_collector import UserIDCollector
from pandas import DataFrame


searchQuery = "corona OR coronavirus OR covid OR covid19 OR covid-19"

uidc = UserIDCollector(searchQuery)
uidc.getUserIds(number_pages=1)
uidc.saveUserData('Users.csv')
userList = uidc.getUserIdList()

print("\n-------------------------------------- | | | | | | | | | | | | | | | | | | | | ------------------------------------------")
print("Total",len(userList)," Users Data is saved")
print("-------------------------------------- | | | | | | | | | | | | | | | | | | | | ------------------------------------------\n")

ttlc = TweetTimeLineCollector(userList)
ttlc.dataCollector()

DataFrame(ttlc.userData).to_csv("UsersData.csv")
