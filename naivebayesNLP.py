import nltk
from nltk.stem import PorterStemmer,WordNetLemmatizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import os

plt.style.use('ggplot')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class NativeBayesNLP:

    def __init__(self, numberOfDFiles, numberOfNDFiles, data_path_d="reddit_depression",
                 data_path_nd="reddit_non_depression"):

        self.data_path_d = data_path_d
        self.data_path_nd = data_path_nd
        self.data_path_d_test = "reddit_depression_testset"
        self.df = pd.DataFrame(columns=['text', 'depression'])
        self.numberOfDFiles = numberOfDFiles
        self.numberOfNDFiles = numberOfNDFiles
        self.depressionClass = 1
        self.nonDepressionClass = 0
        self.classifier = MultinomialNB()
        self.counts = 0
        self.test_counts= 0
        self.featnames = 0


    def preprocessing(self):
        self.checkfilesCounts(self.data_path_d)
        self.checkfilesCounts(self.data_path_nd)
        self.df = self.getTextFromFiles(self.df, self.data_path_d, self.depressionClass, self.numberOfDFiles)
        self.df = self.getTextFromFiles(self.df, self.data_path_nd, self.nonDepressionClass, self.numberOfNDFiles)
        self.dataPreprocessingForX(self.df, 'text')
        self.dataPreprocessingForY(self.df, 'depression')
        X = self.df['text'].values
        Y = self.df['depression'].values
        indices = np.arange(len(X))
        X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(X, Y, indices, test_size=0.15)
        return X_train, X_test, Y_train, Y_test, idx_train, idx_test


    def getTextFromFiles(self, df, data_path, zeroOrOne , limit):
        """Return Data Frame """

        for file in os.listdir(data_path)[:limit]:
            with open(data_path + "/" + file, 'r', encoding="ISO-8859-1") as file1:
                file1 = file1.read()
                df = df.append({'text': file1, 'depression': int(zeroOrOne)}, ignore_index=True)

        return df


    # def getTextFromFiles_Test(self, df_test, data_path, limit):
    #     """Return Data Frame """
    #
    #     for file in os.listdir(data_path)[:limit]:
    #         with open(data_path + "/" + file, 'r', encoding="ISO-8859-1") as file1:
    #             file1 = file1.read()
    #             df_test = df_test.append({'text': file1}, ignore_index=True)
    #
    #     return df_test

    def dataPreprocessingForX(self, df, columnName1):
        df[columnName1] = df[columnName1].map(lambda text: text.lower())
        df[columnName1] = df[columnName1].map(lambda text: nltk.tokenize.word_tokenize(text))
        stop_words = set(nltk.corpus.stopwords.words('english'))
        df[columnName1] = df[columnName1].map(lambda tokens: [w for w in tokens if not w in stop_words])
        df[columnName1] = df[columnName1].map(lambda text: ' '.join(text))
        df[columnName1] = df[columnName1].map(lambda text: re.sub('[^A-Za-z]+', ' ', text))
        df[columnName1] = df[columnName1].map(lambda text: nltk.tokenize.word_tokenize(text))
        lemmatizer = WordNetLemmatizer()
        df[columnName1] = df[columnName1].map(lambda text: [lemmatizer.lemmatize(i) for i in text])
        df[columnName1] = df[columnName1].map(lambda text: ' '.join(text))

    def dataPreprocessingForY(self, df, columnName2):
        df[columnName2] = df[columnName2].astype('int32')

    def checkfilesCounts(self, data_path):
        print("files Counts in "+ data_path +": ", len(os.listdir(data_path)))


    def vectorizingInput(self, ngram_range, min_df):
        pass



    def countBarChart(self):
        # tmp1 = self.counts.toarray()
        # scva = self.featnames
        # scvb = tmp1
        x = self.featnames
        y = self.counts.toarray().sum(axis=0)
        tempDict = {}
        for indx , i in enumerate(x):
            tempDict[i] = y[indx]
        # x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
        tempDict = {k: v for k, v in sorted(tempDict.items(), key=lambda item: item[1])}

        # x_pos = [i for i, _ in enumerate(x)]

        plt.bar(list(tempDict.keys())[:10], list(tempDict.values())[:10])
        plt.xlabel("Words")
        plt.ylabel("Counts")
        plt.title("Training Data Word Counts")
        plt.savefig("TrainingDataWordCounts.png")
        # plt.xticks(x_pos, x)

        plt.show()
    # def saveCountInCSV(self):
    #     scva = count_vectorizer.get_feature_names()
    #     scvb = tmp1.sum(axis=0)

        #
        # np.savetxt("csv1.csv",scva,delimiter=",", fmt="%s")
        # np.savetxt("csv2.csv",scvb,delimiter=",")


    # test_dataset에서 각 원문의 단어가 몇 번 나왔는지 확인 및 추출
    # dump1 =np.array(counts)
    # df_dump1 = pd.DataFrame(columns=['text', "count"])
    # temp = dict()

    # df_test = pd.DataFrame(columns=['text'])

    # df_test = getTextFromFiles_Test(df_test, data_path_d_test,40)
    # print(df_test.shape)
    # dataPreprocessingForX(df_test, 'text')
    # print(df_test.shape)




    # counts = count_vectorizer.fit_transform(df['text'].tolist())


    # print(count_vectorizer.get_feature_names())
    # print(counts.toarray())
    # tmp1 = counts.toarray()
    # # print(tmp1.sum(axis=0))
    # scva = count_vectorizer.get_feature_names()
    # scvb = tmp1.sum(axis=0)
    #
    # np.savetxt("csv23.csv",scva,delimiter=",", fmt="%s")
    # np.savetxt("csv24.csv",scvb,delimiter=",")

    # dump1 = count_vectorizer.get_feature_names()
    # dump2 = counts.toarray()

    # print(counts)
    def fitandPredict(self, X_train, Y_train, X_test, ngram_range=(1, 3), min_df=50):

        count_vectorizer = CountVectorizer(ngram_range = ngram_range, min_df = min_df)
        self.counts = count_vectorizer.fit_transform(X_train)
        self.featnames = count_vectorizer.get_feature_names()
        self.classifier.fit(self.counts, Y_train)
        self.test_counts = count_vectorizer.transform(X_test)
        predictions = self.classifier.predict(self.test_counts)
        print(predictions)

        return predictions

    def score(self, predictions ,Y_test):
        # array1 = np.array(predictions)
        # array2 = np.array(Y_test)
        print ("Accuracy: ",(predictions == Y_test).mean())
        return (predictions == Y_test).mean()



    def addtionalFitandPredictForTFIDF(self,Y_train):
        tfidf_vectorizer = TfidfTransformer().fit(self.counts)
        targets = Y_train
        self.classifier.fit(self.counts, targets)
        example_tfidf = tfidf_vectorizer.transform(self.test_counts)
        predictions_tfidf = self.classifier.predict(example_tfidf)
        print(predictions_tfidf)
        return predictions_tfidf

    # Wordcloud
    def makeWorldCloud(self, idx_test, predictions):
        # Depression
        indexList = []
        for idx_predict, i in enumerate(idx_test):
            if predictions[idx_predict] == 1:
                indexList.append(i)
        # print(type(self.df["text"].iloc[indexList]))
        df_test = self.df["text"].iloc[indexList]
        depression_words = ''.join(list(df_test.tolist()))
        depression_wordclod = WordCloud(width = 512,height = 512).generate(depression_words)
        plt.figure(figsize = (10, 8), facecolor = 'k')
        plt.imshow(depression_wordclod)
        plt.axis('off')
        plt.tight_layout(pad = 0)
        plt.savefig("DepressionWordCloud.png")
        plt.show()

        # Non Depression
        indexList = []
        for idx_predict, i in enumerate(idx_test):
            if predictions[idx_predict] == 0:
                indexList.append(i)
        # print(type(self.df["text"].iloc[indexList]))
        df_test = self.df["text"].iloc[indexList]
        n_depression_words = ''.join(list(df_test.tolist()))
        n_depression_wordclod = WordCloud(width=512, height=512).generate(n_depression_words)
        plt.figure(figsize=(10, 8), facecolor='k')
        plt.imshow(n_depression_wordclod)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig("NonDepressionWordCloud.png")
        plt.show()


if __name__ == "__main__":
    obj = NativeBayesNLP(numberOfDFiles =1000, numberOfNDFiles=500, data_path_d="reddit_depression",
                     data_path_nd="reddit_non_depression")
    X_train, X_test, Y_train, Y_test, idx_train, idx_test = obj.preprocessing()
    predictions = obj.fitandPredict( X_train, Y_train, X_test, ngram_range=(1, 5), min_df=50)
    obj.score(predictions ,Y_test)
    obj.makeWorldCloud(idx_test, predictions)
    obj.countBarChart()
    # predictions2 = obj.addtionalFitandPredictForTFIDF(Y_train)
    # obj.score(predictions2 ,Y_test)