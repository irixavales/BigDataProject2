import json
import time
from twitter import OAuth, TwitterStream
from credentials import credentials as cred

auth = OAuth(cred['token'], cred['token_secret'], cred['consumer_key'], cred['consumer_secret'])

stream = TwitterStream(auth=auth)

# counter for tweets
tweet_count = 0
# total tweets to fetch
total_tweets = 1000
# json file to write tweets to
file_path = 'tweets-from-stream.json'

initial_time = int(round(time.time() * 1000))

for tweet in stream.statuses.sample():
    with open(file_path, 'a') as jsonfile:
        json.dump(tweet, jsonfile)
        jsonfile.write("\n") # Add new line because Py JSON does not

    tweet_count = tweet_count + 1
    if tweet_count % 100 == 0:
        print('tweets streamed: %s' % tweet_count)

    if tweet_count >= total_tweets:
        break

final_time = int(round(time.time() * 1000))
print('\ntotal tweets streamed: %s' % tweet_count)
print('tweets saved in file: %s' % file_path)
print('total run time: %s ms' % (final_time - initial_time))
