import requests
import json

url = 'https://bitcoin-prediction.herokuapp.com/predict'

d=[[144.54 ],
       [139.   ],
       [116.99 ],
       [105.21 ],
       [ 97.75 ],
       [112.5  ],
       [115.91 ],
       [112.3  ],
       [111.5  ],
       [113.566],
       [112.67 ],
       [117.2  ],
       [115.243],
       [115.   ],
       [117.98 ],
       [111.5  ],
       [114.22 ],
       [118.76 ],
       [123.015]]

j_data = json.dumps(d)
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
r = requests.post(url, data=j_data, headers=headers)
print(r, r.text)