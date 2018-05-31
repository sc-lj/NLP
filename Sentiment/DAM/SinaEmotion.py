# coding:utf-8
"""
获取新浪表情
"""
import requests,json

URL='http://api.t.sina.com.cn/emotions.json'

head={'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36',
      'Host':'api.weibo.com'}

def get_sina_emotion(types=['face','ani','cartoon']):
    """
    :param type: "face":普通表情，"ani"：魔法表情，"cartoon"：动漫表情
    :return:
    """
    data={}
    for type in types:
        PARAMS = {'source': '1362404091','type':type}
        content=requests.get(URL,params=PARAMS,headers=head).content.decode('utf-8')
        content=json.loads(content)
        data[type]=content
    content=json.dumps(data)
    with open('./emotion.json','w') as f:
        f.write(content)

# get_sina_emotion()



