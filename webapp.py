"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import os
from PIL import Image

import torch
from flask import Flask, render_template, request, redirect
# from pyngrok import ngrok


app = Flask(__name__)

nameVN = {'As': 'Át bích',
          '1s': '1 bích',
          '2s': '2 bích',
          '3s': '3 bích',
          '4s': '4 bích',
          '5s': '5 bích',
          '6s': '6 bích',
          '7s': '7 bích',
          '8s': '8 bích',
          '9s': '9 bích',
          '10s': '10 bích',
          'Js': 'J bích',
          'Qs': 'Q bích',
          'Ks': 'K bích',
          'Ah': 'Át cơ',
          '1h': '1 cơ',
          '2h': '2 cơ',
          '3h': '3 cơ',
          '4h': '4 cơ',
          '5h': '5 cơ',
          '6h': '6 cơ',
          '7h': '7 cơ',
          '8h': '8 cơ',
          '9h': '9 cơ',
          '10h': '10 cơ',
          'Jh': 'J cơ',
          'Qh': 'Q cơ',
          'Kh': 'K cơ',
          'Ac': 'Át chuồn',
          '1c': '1 chuồn',
          '2c': '2 chuồn',
          '3c': '3 chuồn',
          '4c': '4 chuồn',
          '5c': '5 chuồn',
          '6c': '6 chuồn',
          '7c': '7 chuồn',
          '8c': '8 chuồn',
          '9c': '9 chuồn',
          '10c': '10 chuồn',
          'Jc': 'J chuồn',
          'Qc': 'Q chuồn',
          'Kc': 'K chuồn',
          'Ad': 'Át rô',
          '1d': '1 rô',
          '2d': '2 rô',
          '3d': '3 rô',
          '4d': '4 rô',
          '5d': '5 rô',
          '6d': '6 rô',
          '7d': '7 rô',
          '8d': '8 rô',
          '9d': '9 rô',
          '10d': '10 rô',
          'Jd': 'J rô',
          'Qd': 'Q rô',
          'Kd': 'K rô'
          }


@app.route("/", methods=["GET", "POST"])
def predict():

    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        results = model(img, size=640)
        results.render()  # updates results.imgs with boxes and labels
        # print(results.pandas().xyxy[0]["name"].unique())
        cardsDetected = results.pandas().xyxy[0]["name"].unique()
        cardsDetectedVN = []

        for cardName in cardsDetected:
            cardsDetectedVN.append(nameVN[cardName])
        # print(cardsDetectedVN)

        for img in results.imgs:
            img_base64 = Image.fromarray(img)
            img_base64.save("static/image0.jpg", format="JPEG")
        return render_template("index.html", check=True, cardsDetectedVN=cardsDetectedVN)

    return render_template("index.html", check=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    # Fix known issue urllib.error.HTTPError 403: rate limit exceeded https://github.com/ultralytics/yolov5/pull/7210
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    # force_reload to recache
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    model.eval()

    # url = ngrok.connect(5000).public_url
    # print('Henzy Tunnel URL:', url)
    app.run(host="localhost", port=args.port, debug=False)
