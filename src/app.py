from flask import Flask, request, jsonify
import pandas as pd
from recommenders import recommendByTitle

app = Flask(__name__)

@app.route("/recommend")

def recommend():
    title = request.args.get("title")
    recs = recommendByTitle(title, top_n=5)
    return jsonify({"input":title, "recommendations":recs})

if __name__ ==  "__main__":
    app.run(debug=True) 