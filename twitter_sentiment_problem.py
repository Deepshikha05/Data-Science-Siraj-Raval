import tweepy 
from textblob import TextBlob
import csv

consumer_key= 'CONSUMER-KEY'
consumer_secret= 'COMSUMER-SECRET-KEY'

access_token='ACCESS-TOKEN'
access_token_secret='ACCESS-TOKEN-SECRET'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

fieldnames = ['Tweet', 'Polarity']

writer = csv.DictWriter(open("tweet_data.csv", "w"), fieldnames=fieldnames)
writer.writeheader()

public_tweets = api.search('Terrorism', count = 100, since = '2018-01-01', until = '2018-12-17')

for tweet in public_tweets:
	analysis = TextBlob(tweet.text)
	# import pdb; pdb.set_trace()
	if analysis.sentiment.polarity < 0:
		writer.writerow({
			"Tweet": str(tweet.text),
			"Polarity": "Negative"
		})

	else:
		writer.writerow({
			"Tweet": str(tweet.text),
			"Polarity": "Positive"
		})

