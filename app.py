from flask import Flask, render_template, request
import pickle
import pandas as pd

# create Flask application
app = Flask(__name__)

# read object TfidfVectorizer and model from disk
vec = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    error = None
    if request.method == 'POST':
        # message
        msg = request.form['message']
        msg = pd.DataFrame(index=[0], data=msg, columns=['data'])

        # transform data
        text = vec.transform(msg['data'].astype('U'))
        # model
        result = model.predict(text)

        if result == 0:
            result = "real"
        else:
            result = 'fake'

        # return result
        return render_template('index.html', prediction_value=result)
    else:
        error = "Invalid message"
        return render_template('index.html', error=error)


if __name__ == '__main__':
    app.run(host=('0.0.0.0'), debug=True)
