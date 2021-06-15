# The-edge-of-sentimental-analysis-in-Bitcoin-trading-
This project aims to exploit the edge of sentimental analysis in Bitcoin trading #ScrapingStocktwits #TimeSeriesAnalysis #ARIMA #GrangerCausality #Python.

We try to verify if there is a patent of investors who place their Bitcoin buy/sell orders depending on social media trend. It is intuitively that these types of investor are normally novice or beginners, who base their strategy on sentimental factors. Our approach includes the following steps:

- We scrap all posts/comments in Stocktwits that include the ticker BTC. 
- We try to build our own sentimental lexicon by ranking the frequency of words on the bullish or bearish tweets.
- With the lexicon, we derive the sentimental value of each tweets and turn our data into a time series
- Using own-built timeseries with other NLP python packages like TextBlob, Vader, we can show the correlation of sentimental time series with BTC price.

The research shows sentiments for Bitcoin are all stationary by using 4 methods to calculate.(From Jan to Sep 2019). Bitcoin price experienced a positive trend this year, but return is quite stationary. From VAR model and Granger causality test, we found that return of Bitcoin and sentiment of Bitcoin have bidirectional causality.
