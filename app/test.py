from PIL import Image
from io import BytesIO

import requests

taskId = "eda7dd50638384520135e7163093fb3c"
seed = 666666
imageId = "cloud://prod-0gbpqv7wb25ec7fa.7072-prod-0gbpqv7wb25ec7fa-1315342268/onwON4trZZEpKN44CzLSubSVoy9A/1669563468557PMSbMGHZIpoS00ef837291b8f87ef2ef6e67b8040f85.jpeg"

png = requests.get('https://img-blog.csdnimg.cn/20181106200939520.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xlZW1ib3k=,size_16,color_FFFFFF,t_70')

img = Image.open(BytesIO(png.content)).convert('RGB')


final = BytesIO(png.content)

output = BytesIO()

img.save(output, format='PNG')

print(output)