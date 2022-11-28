import torch
import flask
from diffusers import (
    DiffusionPipeline,
    StableDiffusionImg2ImgPipeline
)

from setting import (
    config,
    hf_token
)

from utils import (
    pil_to_file,
    retrieve_param,
    get_compute_platform,
)


def dummy(images, **kwargs):
    return images, False

# pipe.safety_checker = dummy
# pipeline.enable_attention_slicing()


app = flask.Flask(__name__)

txt2imgPipe = DiffusionPipeline.from_pretrained(
    config['model_id'],
    torch_dtype=torch.float16,
    use_auth_token=hf_token.strip()
).to('cuda')
img2imgPipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    config['model_id'], use_auth_token=hf_token.strip(), torch_dtype=torch.float16).to('cuda')


@app.route('/inference', methods=['POST'])
def draw():
    print('Received Draw')
    return _generate()


@app.route('/ping', methods=['POST'])
def ping():
    print('Received Draw')
    return flask.jsonify({"msg": 'Image Created', "code": 0})


def _generate():
    print('TASK START')
    image = flask.request.form['image']
    generator = torch.Generator(device='cuda')
    seed = generator.seed()
    try:
        args_dict = {
            # TODO sampler
            'prompt': [flask.request.form['prompt']],
            'taskId': flask.request.form['taskId'],
            'num_inference_steps': retrieve_param('num_inference_steps', flask.request.form, int, 50),
            'guidance_scale': retrieve_param('guidance_scale', flask.request.form, float, 7.5),
            'eta': retrieve_param('eta', flask.request.form, float, 0.0),
            'strength': retrieve_param('strength', flask.request.form, float, 0.7)
        }

        if not image:
            pipe = txt2imgPipe
            args_dict['init_image'] = None
            args_dict['width'] = retrieve_param(
                'width', flask.request.form, int, 512)
            args_dict['height'] = retrieve_param(
                'height', flask.request.form, int, 512)
        else:
            pipe = img2imgPipe
            args_dict['init_image'] = image

        print('START RENDER')
        pipeline_output = pipe(**args_dict)

        result = pipeline_output[0]

        print('RENDERED RESULT : ', result)
        pil_to_file(result['image'].convert('RGBA'), args_dict['taskId'], seed)

        return flask.jsonify({
            "msg": 'Image Created',
            "code": 0
        })

    except RuntimeError as e:
        print(str(e))
        return flask.jsonify({"msg": 'GPU Error', "code": -1})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1337, debug=False)
