from PIL import Image
from io import BytesIO

import requests
from utils import (
  upload_wximg,
  getAccessToken
)

token = getAccessToken()
taskId = "eda7dd50638384520135e7163093fb3c"
seed = 1234
imageId = "cloud://prod-0gbpqv7wb25ec7fa.7072-prod-0gbpqv7wb25ec7fa-1315342268/onwON4trZZEpKN44CzLSubSVoy9A/1669563468557PMSbMGHZIpoS00ef837291b8f87ef2ef6e67b8040f85.jpeg"

png = requests.get('https://img-blog.csdnimg.cn/20181106200939520.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xlZW1ib3k=,size_16,color_FFFFFF,t_70')

print(png.content)

img = Image.open(BytesIO(png.content)).convert('RGB')


output = BytesIO()

img.save(output, format='PNG')

upload_wximg(token, taskId, seed, png)