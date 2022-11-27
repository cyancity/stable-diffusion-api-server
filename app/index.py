import torch
import flask
import diffusers
from concurrent.futures import ThreadPoolExecutor

from setting import (
    config,
    hf_token
)

from utils import (
    finishTask,
    retrieve_param,
    getAccessToken,
    get_wximg_by_id,
    get_compute_platform,
    upload_wximg
)

##################################################
# Engines

gpu_running = False
executor = ThreadPoolExecutor(2)

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
        # elif custom_model_path:
        #     self.engine = diffusers.StableDiffusionPipeline.from_pretrained(custom_model_path,
        #                                                                     feature_extractor=sibling.engine.feature_extractor)
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




##################################################
# App
# Initialize app:
app = flask.Flask(__name__)

# Initialize engine manager:

# Add supported engines to manager:
txt2img = EngineStableDiffusion(
    diffusers.StableDiffusionPipeline, sibling=None)
img2img = EngineStableDiffusion(
    diffusers.StableDiffusionImg2ImgPipeline, sibling=txt2img)
masking = EngineStableDiffusion(
    diffusers.StableDiffusionInpaintPipeline, sibling=txt2img)

# @app.route('/custom_models', methods=['GET'])
# def stable_custom_models():
#     if custom_models == None:
#         return flask.jsonify( [] )
#     else:
#         return custom_models


@app.route('/draw', methods=['POST'])
def draw():
    return _generate()

# @app.route('/masking', methods=['POST'])
# def stable_masking():
#     return _generate('masking')

# @app.route('/custom/<path:model>', methods=['POST'])
# def stable_custom(model):
#     return _generate('txt2img', model)

def handleTask(reqBody):
    print('===> Task Start')
    print(reqBody)
    gpu_running = True
    accessToken = getAccessToken()
    # Handle request:
    prompt = reqBody['prompt']
    imageId = reqBody['imageId']
    taskId = reqBody['taskId']

    seed = retrieve_param('seed', reqBody, int, 0)
    count = retrieve_param('num_outputs', reqBody, int,   1)

    try:
        total_results = []
        for i in range(count):
            if (seed == 0):
                generator = torch.Generator(
                    device=get_compute_platform('generator'))
            else:
                generator = torch.Generator(
                    device=get_compute_platform('generator')).manual_seed(seed)

            new_seed = generator.seed()
            args_dict = {
                # sampler
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
                wx_img = get_wximg_by_id(imageId, accessToken)
                if not wx_img:
                    finishTask(accessToken, taskId)
                    gpu_running = False
                    return
                args_dict['init_image'] = wx_img['image']

            args_dict['strength'] = retrieve_param(
                'strength', reqBody, float, 0.7)
            # if (task == 'masking'):
            #     mask_img_b64 = reqBody[ 'mask_image' ]
            #     mask_img_b64 = re.sub( '^data:image/png;base64,', '', mask_img_b64 )
            #     mask_img_pil = b64_to_pil( mask_img_b64 )
            #     args_dict[ 'mask_image' ] = mask_img_pil
            # Perform inference:

            print('===> Start Rendering')
            pipeline_output = pipe.process(args_dict)
            pipeline_output['seed'] = new_seed
            print('===> Rendered')
            gpu_running = False
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

        print('===> prepare upload wximg')
        upload_wximg(accessToken, taskId, images[0]['seed'], images[0]['image'])

    except RuntimeError as e:
        print(str(e))
        gpu_running = False
        finishTask(accessToken, taskId)


def _generate():
    # Prepare output container:
    if gpu_running:
        return flask.jsonify({"msg": 'Gpu busy', "code": -1})
    else:
        executor.submit(handleTask, flask.request.form)

        return flask.jsonify({"msg": 'Gpu Start', "code": 0})




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1337, debug=True)
