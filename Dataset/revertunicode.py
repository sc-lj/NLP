# coding:utf-8

f=open("synonym/哈工大社会计算与信息检索研究中心同义词词林扩展版/词性-词义_合并结果.txt",'r',encoding='gbk')
data=f.read()
f.close()
data=data.split(" ")
print(len(data))

f=open('synonym/POS_WMean.txt','w')
for da in data:
    f.write(da+'\n')
f.close()