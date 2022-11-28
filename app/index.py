import torch
import flask
import diffusers

from setting import (
    config,
    hf_token
)

from utils import (
    retrieve_param,
    get_compute_platform,
)

class Engine(object):
    def __init__(self):
        pass

    def process(self, kwargs):
        return []

def dummy(images, **kwargs):
    return images, False

class EngineStableDiffusion(Engine):
    def __init__(self, pipe, sibling=None, custom_model_path=None):
        super().__init__()
        if sibling == None:
            self.engine = pipe.from_pretrained(
                config['model_id'],
                use_auth_token=hf_token.strip(),
                torch_dtype=torch.float16
            )
        else:
            self.engine = pipe(
                vae=sibling.engine.vae,
                text_encoder=sibling.engine.text_encoder,
                tokenizer=sibling.engine.tokenizer,
                unet=sibling.engine.unet,
                scheduler=sibling.engine.scheduler,
                safety_checker=sibling.engine.safety_checker,
                feature_extractor=sibling.engine.feature_extractor
            )
        pipeline = self.engine.to(get_compute_platform('engine'))

        pipe.safety_checker = dummy
        pipeline.enable_attention_slicing()

    def process(self, kwargs):
        output = self.engine(**kwargs)
        return {'image': output.images[0], 'nsfw': output.nsfw_content_detected[0]}

app = flask.Flask(__name__)


txt2img = EngineStableDiffusion(
    diffusers.StableDiffusionPipeline, sibling=None)
img2img = EngineStableDiffusion(
    diffusers.StableDiffusionImg2ImgPipeline, sibling=txt2img)
masking = EngineStableDiffusion(
    diffusers.StableDiffusionInpaintPipeline, sibling=txt2img)


@app.route('/draw', methods=['POST'])
def draw():
    return handleTask(flask.request.form)

# @app.route('/masking', methods=['POST'])
# def stable_masking():
#     return _generate('masking')

def handleTask(reqBody):
    total_results = []
    prompt = reqBody['prompt']
    taskId = reqBody['taskId']
    imageId = reqBody['imageId']
    seed = retrieve_param('seed', reqBody, int, 0)
    count = retrieve_param('num_outputs', reqBody, int, 1)

    print('===> Task Start', prompt ,taskId)

    try:
        for i in range(count):
            if (seed == 0):
                generator = torch.Generator(
                    device=get_compute_platform('generator'))
            else:
                generator = torch.Generator(
                    device=get_compute_platform('generator')).manual_seed(seed)

            new_seed = generator.seed()
            args_dict = {
                # TODO sampler
                'prompt': [prompt],
                'num_inference_steps': retrieve_param('num_inference_steps', reqBody, int, 50),
                'guidance_scale': retrieve_param('guidance_scale', reqBody, float, 7.5),
                'eta': retrieve_param('eta', reqBody, float, 0.0),
                'generator': generator
            }

            if not imageId:
                print('Get Wx Image Failed')
                pipe = txt2img
                args_dict['init_image'] = None
                args_dict['width'] = retrieve_param(
                    'width', reqBody, int, 512)
                args_dict['height'] = retrieve_param(
                    'height', reqBody, int, 512)
            else:
                pipe = img2img
                # TODO get Wx Image
                # args_dict['init_image'] = wx_img['image']

            args_dict['strength'] = retrieve_param(
                'strength', reqBody, float, 0.7)

            print('===> Start Rendering')
            pipeline_output = pipe.process(args_dict)
            pipeline_output['seed'] = new_seed
            print('===> Rendered')
            total_results.append(pipeline_output)

        # Prepare response
        images = []
        for result in total_results:
            images.append({
                'image': result['image'].convert('RGBA'),
                'seed': result['seed'],
                'mime_type': 'image/png',
                'nsfw': result['nsfw']
            })
        return flask.jsonify({"msg": 'Image Created', "code": 0})


    except RuntimeError as e:
        print(str(e))
        return flask.jsonify({"msg": 'GPU Error', "code": -1})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1337, debug=True)
