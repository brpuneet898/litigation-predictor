from flask import Flask, render_template, request, redirect, url_for
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_case():
    file = request.files.get("case_file")

    if not file:
        return redirect(url_for("index"))

    print(f"Received file: {file.filename}")

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
