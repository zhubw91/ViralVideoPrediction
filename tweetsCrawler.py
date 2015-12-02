from TwitterSearch import *
import time

class TwitterSearchObject:
    tso = None
    ts = None

    def __init__(self):
        try:
            self.tso = TwitterSearchOrder() # create a TwitterSearchOrder object
            #tso.set_keywords([keyword]) # let's define all words we would like to have a look for
            #tso.set_language('de') # we want to see German tweets only
            self.tso.set_include_entities(False) # and don't give us all those entity information

            self.ts = TwitterSearch(
                consumer_key = "5LmBti7QyhAPohaFhZetjkUmM",
                consumer_secret = "ig16K7XHzAhWccNqylZyOrev0c4yRvL6HHhcEQxm423ZKWFLKL",
                access_token = "742676262-2XYoYFxiRDFKDb8y8l0x7tLD9suzRWhqMyDLrXWr",
                access_token_secret = "zw0LWEUonJVNkrNgIBWZUyEcXd02sFvcJtIm7bWjnSzrm"
                )
        except TwitterSearchException as e:
            print(e)

    def get_key_to_tweetsNum(self,keyword):
        tweetNum = 0
        try:
            self.tso.set_keywords([keyword]) # let's define all words we would like to have a look for

            def my_callback_closure(current_ts_instance): # accepts ONE argument: an instance of TwitterSearch
                    queries, tweets_seen = current_ts_instance.get_statistics()
                    if queries > 0 and (queries % 5) == 0: # trigger delay every 5th query
                            time.sleep(60) # sleep for 60 seconds

            tweetNum =  self.ts.search_tweets_iterable(self.tso,callback=my_callback_closure).get_statistics()[1]

            # for tweet in ts.search_tweets_iterable(tso):
            #     print( '@%s tweeted: %s' % ( tweet['user']['screen_name'], tweet['text'] ) )

        except TwitterSearchException as e:
            print(e)
        return tweetNum


