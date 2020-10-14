from flask import Flask,render_template,url_for, request
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import warnings
warnings.filterwarnings("ignore")

def remove_pattern(input_txt,pattern):
    r = re.findall(pattern,input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)
    return input_txt

app = Flask(__name__)
data=pd.read_csv('train.csv')

data=data.drop(['id'],axis=1)

data['Tweets'] = np.vectorize(remove_pattern)(data['tweet'],"@[\w]*")
data['Tweets'] = data['Tweets'].str.replace('[^a-zA-Z]'," ")

tokenized_tweet = data['Tweets'].apply(lambda x: x.split())
pstem = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x:[pstem.stem(i) for i in x])

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
data['Tweets'] = tokenized_tweet

from sklearn.utils import resample
majority = data[data.label==0]
minority = data[data.label==1]
 
minority_upsampled = resample(minority, 
                                 replace=True,
                                 n_samples=len(majority),  
                                 random_state=123)
 
data = pd.concat([majority, minority_upsampled])

x_train, x_test, y_train, y_test = train_test_split(data['Tweets'], data['label'], test_size=0.3, random_state=0)

rf=RandomForestClassifier()

cv = CountVectorizer(stop_words='english')

train=cv.fit_transform(x_train)
test=cv.transform(x_test) 
rf.fit(train,y_train)
predicted=rf.predict(test)

confusion_matrix(y_test, predicted)
accuracy_score(predicted,y_test)

@app.route('/')
def home():
	return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
	if request.method=='POST':
		message=request.form['message']
		data=[message]
		df=pd.DataFrame(cv.transform(data).toarray())

		my_prediction = rf.predict(df)
	return render_template('result.html',prediction=my_prediction)

if __name__ == "__main__":
	app.run(port=8000,debug=True)