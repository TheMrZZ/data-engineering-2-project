from flask import Flask, request, jsonify

from classifier import predict

app = Flask(__name__)


@app.route('/classify_sentence', methods=['POST'])
def classify_sentence():
    text = request.json['sentence']

    return jsonify({
        "sentiment": predict(text)
    })


if __name__ == '__main__':
    app.run()
