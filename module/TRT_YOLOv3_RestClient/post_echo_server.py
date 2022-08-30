from flask import Flask, request
app = Flask(__name__)

@app.route("/", methods=['POST'])
def webhook():
    print (request.headers)
    print ("body: %s" % request.get_data())
    return request.get_data()

if __name__ == "__main__":
    app.run()