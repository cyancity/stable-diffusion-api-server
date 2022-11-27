import flask
import base64
import requests
import json
from PIL import Image
from io import BytesIO
from setting import (config)


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

# WX


def get_wximg_by_id(id: str, accessToken: str | bool):
    if accessToken:
        imgPath = config['wx_prefix'] + config['wx_download_path']
        result = requests.post(imgPath, params={
            "access_token": accessToken
        }, json={
            "env": config['wx_img_env'],
            "file_list": [{"fileid": id, "max_age": int(config['img_max_age'])}]
        })
        print('get wx img result', result.json())
        print(result.request.body)
        if result.ok:
            file_id = result.json()["file_list"][0]['fileid']
            download_url = result.json()["file_list"][0]['download_url']
            image = requests.get(download_url)
            print(download_url)
            print(image)
            if image.status_code == 200:
                return {
                    "file_id": file_id,
                    "image": Image.open(BytesIO(image.content))
                }
    return False


def getAccessToken() -> str:
    result = requests.get(config['wx_prefix'] + config['wx_token_path'], {
        "grant_type": 'client_credential',
        "appid": config['app_id'],
        "secret": config['app_secret']
    })

    if result.ok:
        print('getToken', result.json()['access_token'])
        return result.json()['access_token']
    return ''


def finishTask(accessToken: str, taskId: str, fileId: str = ''):
    status = False
    if fileId:
        status = True
    print('===> Finish Task', status)

    r = requests.post(config['wx_prefix'] + config['wx_cloud_func_path'],
        params={
            "access_token": accessToken,
            "env": config['wx_img_env'],
            "name": "finishTask"},
        json={
            "taskId": taskId,
            "success": status,
            "fileId": fileId
        })
    print('===> request detail', r.request.url, r.request.body)
    print('===> Cloud Func Res', r.status_code, r._content)

def upload_wximg(accessToken: str, taskId: str, seed: int, file):
    path = taskId + "/" + "seed" + str(seed) + ".png"
    upload_info = requests.post(config['wx_prefix'] + config['wx_upload_path'], params={
        "access_token": accessToken,
        "env": config['wx_img_env'],
        "path": path
    }).json()

    if not upload_info["errcode"] == 0:
        # Failed
        return finishTask(accessToken, taskId)
    upload_img = requests.post(upload_info['url'], files=file, json={
        "key": path,
        "Signature": upload_info['authorization'],
        "x-cos-security-token": upload_info['token'],
        "x-cos-meta-fileid	": upload_info['cos_file_id']
    })

    if upload_img.ok:
        finishTask(accessToken, taskId, upload_info['file_id'])
    else:
        finishTask(accessToken, taskId)