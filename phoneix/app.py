from newspaper import Article
import streamlit as st
import pandas as pd


from datetime import date

import yfinance as yf
from plotly import graph_objs as go

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import re
import matplotlib.pyplot as plt
import seaborn as sns

from random import randint
import nltk
nltk.download('stopwords')







import nltk





def main():


    st.title("Stock Insights")
    menu = ["Home", "Website", "Indicator"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":

        def user_input_features(text_input):

            df = pd.read_csv(r'C:\Users\pranj\PycharmProjects\phoneix\Combined_News_DJIA.csv', encoding='ISO-8859-1')
            df.dropna(inplace=True)

            df_copy = df.copy()
            df_copy.reset_index(inplace=True)
            train = df_copy[df_copy['Date'] < '20150101']
            test = df_copy[df_copy['Date'] > '20141231']
            y_train = train['Label']
            train = train.iloc[:, 3:28]
            y_test = test['Label']
            test = test.iloc[:, 3:28]

            train.replace(to_replace='[^a-zA-Z]', value=' ', regex=True, inplace=True)
            test.replace(to_replace='[^a-zA-Z]', value=' ', regex=True, inplace=True)

            new_columns = [str(i) for i in range(0, 25)]
            train.columns = new_columns
            test.columns = new_columns
            for i in new_columns:
                train[i] = train[i].str.lower()
                test[i] = test[i].str.lower()

            train_headlines = []
            test_headlines = []
            for row in range(0, train.shape[0]):
                train_headlines.append(' '.join(str(x) for x in train.iloc[row, 0:25]))
            for row in range(0, test.shape[0]):
                test_headlines.append(' '.join(str(x) for x in test.iloc[row, 0:25]))

            ps = PorterStemmer()
            train_corpus = []
            for i in range(0, len(train_headlines)):
                words = train_headlines[i].split()
                words = [word for word in words if word not in set(stopwords.words('english'))]
                words = [ps.stem(word) for word in words]
                headline = ' '.join(words)
                train_corpus.append(headline)

            test_corpus = []
            for i in range(0, len(test_headlines)):
                words = test_headlines[i].split()
                words = [word for word in words if word not in set(stopwords.words('english'))]
                words = [ps.stem(word) for word in words]
                headline = ' '.join(words)
                test_corpus.append(headline)

            down_words = []
            for i in list(y_train[y_train == 0].index):
                down_words.append(train_corpus[i])
            up_words = []
            for i in list(y_train[y_train == 1].index):
                up_words.append(train_corpus[i])

            from sklearn.feature_extraction.text import CountVectorizer
            cv = CountVectorizer(max_features=10000, ngram_range=(2, 2))
            X_train = cv.fit_transform(train_corpus).toarray()
            X_test = cv.transform(test_corpus).toarray()

            import pickle

            with open(r"C:\Users\pranj\PycharmProjects\phoneix\prediction.pkl", "rb") as file:
                model = pickle.load(file)

            def stock_prediction(sample_news):
                sample_news = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sample_news)
                sample_news = sample_news.lower()
                sample_news_words = sample_news.split()
                sample_news_words = [word for word in sample_news_words if not word in set(stopwords.words('english'))]
                ps = PorterStemmer()
                final_news = [ps.stem(word) for word in sample_news_words]
                final_news = ' '.join(final_news)
                temp = cv.transform([final_news]).toarray()
                return model.predict(temp)

            sample_test = df_copy[df_copy['Date'] > '20141231']
            sample_test.reset_index(inplace=True)
            sample_test = sample_test['Top1']

            row = randint(0, sample_test.shape[0] - 1)
            sample_news1 = text_input
            print('News: {}'.format(sample_news1))
            if stock_prediction(sample_news1):
                st.error('Prediction: The stock price will remain the same or will go down.')
            else:
                st.success('Prediction: The stock price will go up!')


            #st.sidebar.add_rows

        text = st.text_input('Please enter text here ')

        if(text):
            user_input_features(text)


            ##st.sidebar.add_rows








    elif choice == "Website":
        #add_page_visited_details("Monitor", datetime.now())
        st.subheader("website App")

        st.subheader("Url")

        url = st.text_input('Please enter Url here ')

        if url:
            article = Article(url, language="en")  # en for English

            article.download()

            article.parse()

            # To perform natural language processing ie..nlp
            article.nlp()

            st.subheader("Article's Summary")

            st.success(article.summary)

            st.subheader("Article's Keywords:")
            st.write(article.keywords)

            from flair.models import TextClassifier
            from flair.data import Sentence

            classifier = TextClassifier.load('en-sentiment')
            sentence = Sentence(article.title)
            classifier.predict(sentence)

            # print sentence with predicted labels
            print('Sentence above is: ', sentence.labels)

            classifier = TextClassifier.load('en-sentiment')
            summary = Sentence(article.summary)
            classifier.predict(summary)

            score = sentence.score * 100



            answer = sentence.to_dict()

            #st.subheader("Prediction")
            #st.success(answer['all labels'][0]['value'])

            if score < 70 :
                st.subheader("Prediction")
                st.warning("HOLD")

                st.subheader("prediction confidence ")
                st.warning(score)

            elif answer['all labels'][0]['value'] == "NEGATIVE":
                st.subheader("Prediction")
                st.error("SELL")
                st.subheader("prediction confidence ")
                st.error(score)

            else :
                st.subheader("Prediction")
                st.success("BUY")
                st.subheader("prediction confidence ")
                st.error(score)












    else:
        st.subheader("Indicator")

        START = "2013-01-01"
        TODAY = date.today().strftime("%Y-%m-%d")

        st.title('Stock Price History')

        selected_stock = st.text_input("Enter Stock Ticker 👇", "Type here...")

        if (st.button('Submit')):

            @st.cache
            def load_data(ticker):
                data = yf.download(ticker, START, TODAY)
                data.reset_index(inplace=True)
                return data

            data = load_data(selected_stock)

            st.subheader('Raw data')
            st.write(data.tail())

            def plot_raw_data():
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
                fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
                st.plotly_chart(fig)

            plot_raw_data()

            df_train = data[['Date', 'Close']]
            df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})




if __name__ == '__main__':
	main()



