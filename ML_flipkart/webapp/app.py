from flask import Flask, render_template, request
import sklearn
import joblib
from clean import clean
import pandas as pd

reviews= pd.read_csv('review_record.csv')
review_vibe=list(reviews['0'])
binary=[0,1]


app=Flask(__name__)

##################

@app.route('/')
def home():

    return render_template('home.html')

@app.route('/prediction', methods = ['GET','POST'])
def predict():
    review = request.form.get('review_predict')
    count=[]
    rev=[]
    if (review != None) and (review != ''):
        review_clean = clean(review)
        model =joblib.load("best_models_word2vec/logistic_regression.pkl")
        pred=model.predict([review_clean])
        for y in pred:
            review_vibe.append(y)
        record = pd.DataFrame(review_vibe)
        record.to_csv('review_record.csv', index=False)
        
        for x in binary:
            if (x == 0) and (x in review_vibe):
                num = review_vibe.count(x)
                count.append(num)
                rev.append('Negative')
            elif (x == 1) and (x in review_vibe):
                num = review_vibe.count(x)
                count.append(num)
                rev.append('Positive')
        return render_template('home.html', pred=pred, count=count, rev=rev)
    else:
        for x in binary:
            if (x == 0) and (x in review_vibe):
                num = review_vibe.count(x)
                count.append(num)
                rev.append('Negative')
            elif (x == 1) and (x in review_vibe):
                num = review_vibe.count(x)
                count.append(num)
                rev.append('Positive')
        return render_template('home.html', count = count, rev=rev )




#################

if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0' )
