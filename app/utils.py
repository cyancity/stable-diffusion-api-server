import flask
import base64
from PIL import Image
from io import BytesIO


def retrieve_param(key, data, cast, default):
    if key in data:
        value = flask.request.form[key]
        value = cast(value)
        return value
    return default


def pil_to_b64(input):
    buffer = BytesIO()
    input.save(buffer, 'PNG')
    output = base64.b64encode(buffer.getvalue()).decode(
        'utf-8').replace('\n', '')
    buffer.close()
    return output


def file_to_pil_image(input):
    output = Image.open(BytesIO(base64.b64decode(input)))
    return output


def get_compute_platform(context):
    return 'cuda'


def pil_to_file(input, taskId, seed):
    buffer = BytesIO()
    input.save('./output/' + taskId + 'seed' + seed, 'PNG')
