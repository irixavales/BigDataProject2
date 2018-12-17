import json
import time
from twitter import OAuth, TwitterStream
from conf.credentials import credentials as cred



class Stream():

    def __init__(self):
        pass

    def stream_tweets(self, total_tweets, output_path='tweets-from-stream.json', filter=[]):
        # authentication
        auth = OAuth(cred['token'], cred['token_secret'], cred['consumer_key'], cred['consumer_secret'])
        # stream twitter object
        stream = TwitterStream(auth=auth)
        # counter for tweets
        tweet_count = 0

        initial_time = int(round(time.time() * 1000))

        for tweet in stream.statuses.filter(track=filter):
            with open(output_path, 'a') as jsonfile:
                json.dump(tweet, jsonfile)
                jsonfile.write("\n")  # Add new line because Py JSON does not

            tweet_count = tweet_count + 1
            print(tweet_count)
            # if tweet_count % 100 == 0:
            #     print('tweets streamed: %s' % tweet_count)

            if tweet_count >= total_tweets:
                break

        final_time = int(round(time.time() * 1000))

        print('\ntotal tweets streamed: %s' % tweet_count)
        print('tweets saved in file: %s' % output_path)
        print('total run time: %s ms' % (final_time - initial_time))


if __name__ == '__main__':
    keywords = 'flu, zika, diarrhea, ebola, headache, measles'
    Stream().stream_tweets(6190, output_path='disease-tweets.json', filter=keywords)
