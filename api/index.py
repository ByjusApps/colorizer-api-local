import tempfile
from datetime import datetime as dt

import cv2
import imutils
from colorizer import colorize
from consts import *
from flask import Flask, jsonify, request, send_file, url_for
from flask_cors import cross_origin
from style_transferer import transfer_style

app = Flask(__name__)
temp_images = []


@app.route("/")
@cross_origin()
def home():
    return "Home"


@app.route("/api/colorizer/")
@cross_origin()
def route_colorizer():
    imageUrl = request.args.get("imageUrl", type=str)
    imageStyle = request.args.get("imageStyle", type=str)

    img = imutils.url_to_image(imageUrl)

    if imageStyle in STYLES_MAP:
        img = transfer_style(img, STYLES_MAP[imageStyle])
        operation_type = "style_transfer"
    else:
        img = colorize(img)
        operation_type = "colorizer"

    timestamp = int(dt.now().timestamp() * 1_000_000)

    if len(temp_images) >= 5:
        temp_images.pop(0)

    temp_images.append((timestamp, img))

    return jsonify(
        {
            "operation_type": operation_type,
            "message": " operation success",
            "image_url": imageUrl,
            "output_url": f"{url_for('route_colorizer_temp',timestamp=timestamp, _external=True)}.jpg",
        }
    )


@app.route("/api/colorizer/temp/<timestamp>")
@cross_origin()
def route_colorizer_temp(timestamp):
    try:
        timestamp = timestamp.replace(".jpg", "")

        img = list(filter(lambda x: x[0] == int(timestamp), temp_images))
        img = img[0][1]

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_file.name, img)

        return send_file(temp_file.name)

    except IndexError:
        return "Image not found"


@app.route("/api/colorizer/temp/list")
@cross_origin()
def route_colorizer_temp_list():
    return jsonify(list(map(lambda x: x[0], temp_images)))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
