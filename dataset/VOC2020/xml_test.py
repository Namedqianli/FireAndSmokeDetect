import os
from xml.dom import minidom

xml_path = './Annotations'
xmls = os.listdir(xml_path)
rate = {} # 创建一个字典用于存放标签名和对应的出现次数

train_file = open('train.txt', 'w')
test_file = open('test.txt', 'w')

print(len(xmls))
count = 0
for xml_file in xmls:
    flag = False
    if xml_file.endswith('.xml'):
        fullname = os.path.join(xml_path,xml_file)
        dom = minidom.parse(fullname) # 打开XML文件
        collection = dom.documentElement # 获取元素对象
        objectlist = collection.getElementsByTagName('object') # 获取标签名为object的信息
        for object in objectlist:
            namelist = object.getElementsByTagName('name') # 获取子标签name的信息
            if len(namelist) == 0:
                print(xml_file)
                continue
            objectname = namelist[0].childNodes[0].data # 取到name具体的值
            if objectname not in rate: # 判断字典里有没有标签，如无添加相应字段
                rate[objectname] = 0
            rate[objectname] += 1
            flag = True

    if flag:
        filename, extension = os.path.splitext(xml_file)
        jpg_path = os.path.join('./JPEGImages', filename + '.jpeg')
        if os.path.exists(jpg_path) == True:
            if count % 100 == 0:
                test_file.write(filename + '\n')
            else:
                train_file.write(filename + '\n')
    count += 1
train_file.close()

print(rate)
# 画图
rate = sorted(rate.items(), key=lambda x: x[1],reverse = True)
import matplotlib.pyplot as plt
object = []
number = []
for key in rate:
    object.append(key[0])
    number.append(key[1])
plt.figure()
plt.bar(object, number)
plt.title('result')
plt.show()
