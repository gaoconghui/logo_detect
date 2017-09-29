import cv2
import flask
import numpy as np
from cv2.cv import CV_IMWRITE_JPEG_QUALITY
from flask import Flask, request, Response
from flask_script import Manager

from logo_detect.delogo import inpaint
from logo_detect.template_detect import detect

app = Flask(__name__)
manager = Manager(app)


def get_np_array_from_tar_object(stream):
    '''converts a buffer from a tar file in np.array'''
    return np.asarray(
        bytearray(stream)
        , dtype=np.uint8)


@app.route('/', methods=['POST'])
def logo_detect():
    result = {"status": "success"}
    try:
        img = cv2.imdecode(get_np_array_from_tar_object(request.files["image"].stream.read()), 0)
        name, location = detect(img)
        if name:
            data = {
                "name": name,
                "left_top": list(location[0]),
                "right_bottom": list(location[1]),
                "image_shape": img.shape[::-1]
            }
            result["data"] = data
    except:
        import traceback
        print traceback.format_exc()
        result['status'] = "failed"
    return flask.jsonify(result)


@app.route('/inpaint', methods=['POST'])
def logo_inpaint():
    result = {"status": "success"}
    try:
        image_nparray = get_np_array_from_tar_object(request.files["image"].stream.read())
        img_gary = cv2.imdecode(image_nparray, 0)
        name, location = detect(img_gary)
        if location:
            img = cv2.imdecode(image_nparray, 1)
            dst = inpaint(img, location)
            return Response(np.array(cv2.imencode(".jpeg", dst, [int(CV_IMWRITE_JPEG_QUALITY), 90])[1]).tobytes(),
                            mimetype="image/jpeg")
    except:
        import traceback
        error_msg = traceback.format_exc()
        print error_msg
        result['status'] = "failed"
        result['error_msg'] = error_msg
    return flask.jsonify(result)


if __name__ == '__main__':
    manager.run()
