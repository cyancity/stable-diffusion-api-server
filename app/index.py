import os
import torch
import flask

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline
)

from setting import (
    config,
    hf_token
)

from utils import (
    iso_date_time,
    retrieve_param,
    skip_safety_checker
)

###############################
# Engine


class Engine(object):
    def __init__(self) -> None:
        pass

    def process(selef, kwargs):
        return []


class EngineStableDiffusion(Engine):
    def __init__(self, pipe, sibling=None) -> None:
        super().__init__()
        print("load pipeline start:", iso_date_time(), flush=True)
        if sibling == None:
            self.engine = pipe.from_pretrained(
                config['model_id'],
                revision="main",
                torch_dtype=torch.float16,
                use_auth_token=hf_token.strip()
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
        self.engine.to('cuda')

        if bool(config['skip']):
            self.engine.safety_checker = skip_safety_checker
        if bool(config['attention_slicing']):
            self.engine.enable_attention_slicing()
        print('attention_slicing', bool(config['attention_slicing']))
        print(self.engine)
        print("loaded models after:", iso_date_time(), flush=True)

    def process(self, kwargs):
        return self.engine(**kwargs)


class EngineManager(object):
    def __init__(self) -> None:
        self.engines = {}

    def has_engine(self, name):
        return (name in self.engines)

    def add_engine(self, name, engine):
        if self.has_engine(name):
            return False
        self.engines[name] = engine
        return True

    def get_engine(self, name):
        if not self.has_engine(name):
            return None
        return self.engines[name]
###############################


app = flask.Flask(__name__)

manager = EngineManager()

manager.add_engine('txt2img', EngineStableDiffusion(
    StableDiffusionPipeline, sibling=None))
manager.add_engine('img2img', EngineStableDiffusion(
    StableDiffusionImg2ImgPipeline, sibling=manager.get_engine('txt2img')))

IMG_TASK = 'img2img'
TXT_TASK = 'txt2img'


@app.route('/inference', methods=['POST'])
def draw():
    print('Received Draw')
    return _generate()


def _generate():
    task = flask.request.form['task']
    print('TASK START', task)

    engine = manager.get_engine(task)

    generator = torch.Generator(device='cuda')
    seed = generator.seed()
    try:
        seed = retrieve_param('seed', flask.request.form, int, 0)
        if (seed == 0):
            generator = torch.Generator(device='cuda')
        else:
            generator = torch.Generator(device='cuda').manual_seed(seed)

        new_seed = generator.seed()
        prompt = flask.request.form['prompt'].replace(" ", "_")[:170]
        taskId = flask.request.form['taskId']
        args_dict = {
            # TODO sampler
            'prompt': [prompt],
            'num_inference_steps': retrieve_param('num_inference_steps', flask.request.form, int, 50),
            'guidance_scale': retrieve_param('guidance_scale', flask.request.form, float, 7.5),
            'eta': retrieve_param('eta', flask.request.form, float, 0.0),
            'generator': generator
        }

        if (task == TXT_TASK):
            args_dict['width'] = retrieve_param(
                'width', flask.request.form, int, 512)
            args_dict['height'] = retrieve_param(
                'height', flask.request.form, int, 512)
        if (task == IMG_TASK):
            # TODO blob to pil
            init_image = flask.request.form['init_image']
            args_dict['init_image'] = init_image
            args_dict['strength'] = retrieve_param(
                'strength', flask.request.form, float, 0.7)

        print('START RENDER')
        pipeline_output = engine.process(args_dict)
        print('RENDERED OUTPUT : ', pipeline_output)

        result = pipeline_output[0]

        # save to local file
        for i, img in enumerate(result.images):
            out = f"{taskId}_seed{new_seed}_steps{args_dict['num_inference_steps']}.png"
            img.save(os.path.join("output", out))

        return flask.jsonify({
            "msg": 'Image Created',
            "code": 0,
            "data": {
                "seed": seed,
                "imagePath": '',
                "taskId": taskId,
            }
        })

    except RuntimeError as e:
        print(str(e))
        return flask.jsonify({"msg": 'GPU Error', "code": -1})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1337, debug=True)
