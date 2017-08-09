import cv2
import flask
import numpy as np
from flask import Flask, request

from logo_detect.template_detect import detect

app = Flask(__name__)


def get_np_array_from_tar_object(stream):
    '''converts a buffer from a tar file in np.array'''
    return np.asarray(
        bytearray(stream)
        , dtype=np.uint8)


@app.route('/', methods=['POST'])
def logo_detect():
    result = {"status": "success"}
    try:
        print request.files
        img = cv2.imdecode(get_np_array_from_tar_object(request.files["image"].stream.read()), 0)
        name, location = detect(img)
        if name:
            data = {
                "name": name,
                "left_top": list(location[0]),
                "right_bottom": list(location[1])
            }
            result["data"] = data
    except:
        import traceback
        print traceback.format_exc()
        result['status'] = "failed"
    return flask.jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
