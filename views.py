import os
from flask import Flask, session, request, render_template, flash, redirect, url_for, send_from_directory
from infer import Inference
from config import SAVE_IMAGE_DIR, ROOT_DIR
from PIL import Image
from loguru import logger
import datetime

app = Flask(__name__)
model = Inference()

@app.route("/")
def index():
    return render_template("index.html")



@app.route("/login", methods=["GET", "POST"])
def login():
    try:
        session.pop("username")
    except:
        pass

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        
        if (username == "admin") and (password == "1234"):
            session["username"] = username
            flash("Logged in successfully")
            return render_template("index.html")
            
        else:
            flash("Either username or password is incorrect")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("username")
    flash("Successfully logged out")
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        logger.info("Reading Image")
        file = request.files["file"]
        # filename = "cs776a-image.jpg"
        filename = str(datetime.datetime.now())+".jpg"
        # filename = file.filename
        input_path = os.path.join("/tmp/", filename)
        logger.info("Saving Image")
        file.save(input_path)
        logger.info("Saved Image successfully")
        
        logger.info("Resizing Image")
        img = Image.open(input_path)
        img = img.resize((400, 400))
        img.save(input_path)

        logger.info("Correcting orientation")
        model_name = request.form["model"].lower()
        # predict_path = os.path.join(SAVE_IMAGE_DIR, filename)
        model.predict(model_name, input_path)
        return render_template("results.html", filename=filename)

    return render_template("predict.html")


@app.route('/input/<path:filename>')
def get_input_images(filename):
    return send_from_directory(SAVE_IMAGE_DIR,
                               filename, as_attachment=True, cache_timeout=0)


@app.route('/pred/<path:filename>')
def get_pred_images(filename):
    return send_from_directory(SAVE_IMAGE_DIR,
                               "pred_"+filename, as_attachment=True, cache_timeout=0)



@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

