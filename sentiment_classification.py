from SentimentNetwork import SentimentNetwork
from text_processing import clean_text, read_and_preprocess_data, data_extraction
from flask import Flask, request, jsonify, render_template

reviews, labels, vocab, word2index = read_and_preprocess_data()
reviews = reviews.tolist()
labels = labels.tolist()

# initialize and train network
mlp = SentimentNetwork(reviews[:-22000],labels[:-22000], vocab, word2index)
mlp.train(reviews[:-22000],labels[:-22000])
# test network
print('\n')
mlp.test(reviews[-22000:],labels[-22000:])
print('\n')

   
url = 'https://www.skroutz.gr/s/23272388/Xiaomi-Redmi-Note-9-Pro-128GB-Tropical-Green.html#reviews'
rselector= '#sku_reviews_list > li div.review-body'
sselector = '#sku_reviews_list > li div.actual-rating > span'
data = data_extraction(url,rselector,sselector)
data_reviews = data['reviews'].tolist()
data_labels = data['sentiment'].tolist()
print('Accuracy on reviews from website:')
mlp.test(data_reviews, data_labels)
print('\n')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    review = clean_text(review)
    prediction = mlp.run(review)
    
    if prediction == 1:
        return render_template('index.html', prediction_text = 'Το νευρωνικό δίκτυο κατηγοριοποίησε την κριτική ως θετική')
    else:
        return render_template('index.html', prediction_text = 'Το νευρωνικό δίκτυο κατηγοριοποίησε την κριτική ως αρνητική')



if __name__ == '__main__':
    app.run()

    
