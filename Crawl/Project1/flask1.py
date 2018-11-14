import flask

app = flask.Flask(__name__)


@app.route("/")
def hello():
    return "你好"


@app.route("/hi")
def hi():
    return "Hi,你好"


if __name__=="__main__":
    app.run()