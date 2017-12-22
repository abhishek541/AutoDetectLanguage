import os

from sklearn import feature_extraction
from sklearn import linear_model
from sklearn import pipeline
from sklearn import metrics

class DetectLanguage:

    lang_model = None
    languages = []
    confusion_matrix = None

    def getTextData(self, type):
        directory = 'data/' + type
        X = []
        y = []
        files = [lang for lang in os.listdir(directory)]
        self.languages = [lang.split('.')[0] for lang in files]
        for nf, file_name in enumerate(files):
            file_path = os.path.join(directory, file_name)
            with open(file_path, encoding="utf8") as text_file:
                text = text_file.readlines()
                y += [nf] * len(text)
                X += text
        return X, y

    def trainmodel(self):
        Xtr, ytr = self.getTextData('train')
        vectorizer = feature_extraction.text.TfidfVectorizer(ngram_range=(1, 6), analyzer='char')

        self.lang_model = pipeline.Pipeline([('vectorizer', vectorizer), ('clf', linear_model.LogisticRegression())])
        self.lang_model.fit(Xtr, ytr)

    def predict(self):
        Xts, yts = self.getTextData('test')
        yhat = self.lang_model.predict(Xts)
        self.metrics = metrics.classification_report(yts, yhat, target_names=self.languages)

        for text, n in zip(Xts, yhat):
            print(u'{} ==========> language is {}'.format(text, self.languages[n]))

        return Xts, yhat