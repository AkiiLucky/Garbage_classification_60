# -*- coding:utf-8 -*-
from flask import Flask, render_template, request, redirect, url_for
from flask import make_response, json, jsonify
from flask import abort
import numpy as np
import cv2
import torch
from torchvision import models
from torch import nn
import torchvision.transforms as transforms



app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
#app.config['SECRET_KEY'] = '123'

'''
@app.route('/index1/<path:p>', methods=['get' ,'post'])
def index1(p):
    if p == 'image/0':
        return '可回收物'
    return render_template('index.html')

@app.route('/index2', methods=['get' ,'post'])
def index2():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        image_path = request.form.get('name')
        image_pat = request.form.get('password')
        print(image_path, image_pat)
        return 'this is post'

# 重定向
@app.route('/index3')
def index3():
    return redirect('https://www.baidu.com')

@app.route('/index4')
def index4():
    return redirect(url_for('index5'))

@app.route('/index5')
def index5():
    return redirect('https://www.sougou.com')

# *******************************************************
# 返回json格式数据给前端
@app.route('/index6')
def index6():
    data = {
        '一级分类': '可回收物',
        '二级分类': '塑料',
        '相似度': 0.66
    }
    response = make_response(json.dumps(data, ensure_ascii=False))
    response.mimetype = 'application/json'
    return response

@app.route('/index7')
def index7():
    data = {
        '一级分类': '可回收物',
        '二级分类': '塑料',
        '相似度': 0.66
    }
    return jsonify(data)
# *******************************************************

# 在网页抛出异常
@app.route('/index8')
def index8():
    abort(404)
    return 'error'

# 返回图片
@app.route('/index9')
def index9():
    return render_template('index9.html')

# 模板jinja2
@app.route('/index10')
def index10():
    data = {
        '一级分类': '可回收物',
        '二级分类': ['塑料', '易拉罐', '玻璃'],
        '相似度': [0.66, 0.32, 0.22]
    }
    return render_template('index10.html', data=data)


# 自定义过滤器
def maxVal(data):
    return max(data)

app.add_template_filter(maxVal, 'maxVal')

# 定义表单模型类
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, EqualTo
from flask_wtf import FlaskForm

class Register(FlaskForm):
    img_path = StringField(label='图片路径', validators=[DataRequired('图片路径不能为空')])
    submit = SubmitField(label='提交')

@app.route('/register',  methods=['get' ,'post'])
def register():
    form = Register()
    # if form.validate_on_submit():
    return render_template('register.html', form=form)
    # else:
    #     print('提交失败')


# 使用数据库***************************************************
from flask_sqlalchemy import SQLAlchemy
# ************************************************************

'''




def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

net = models.resnet18(pretrained=False)
# 修改全连接层的输出
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 60)
net.load_state_dict(torch.load('resnet18.pkl', map_location='cpu'))
device = torch.device("cpu")
net = net.to(device)
transform = transforms.Compose([
    transforms.ToTensor(),                                                  # 归一化
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),     # 标准化
])
count = 0
labelIndex_4 = {0: '可回收垃圾', 1: '有害垃圾', 2: '厨余垃圾', 3: '其他垃圾'}
labelIndex_60 = {0: '传单', 1: '充电宝', 2: '包', 3: '塑料玩具', 4: '塑料碗盆', 5: '塑料衣架', 6: '快递纸袋', 7: '报纸', 8: '插头电线', 9: '旧书', 10: '旧衣服', 11: '易拉罐', 12: '杂志', 13: '枕头', 14: '毛绒玩具', 15: '泡沫塑料', 16: '洗发水瓶', 17: '牛奶盒等利乐包装', 18: '玻璃', 19: '玻璃瓶罐', 20: '皮鞋', 21: '砧板', 22: '纸板箱', 23: '调料瓶', 24: '酒瓶', 25: '金属食品罐', 26: '锅', 27: '食用油桶', 28: '饮料瓶', 29: '干电池', 30: '废弃水银温度计', 31: '废旧灯管灯泡', 32: '杀虫剂容器', 33: '电池', 34: '软膏', 35: '过期药物', 36: '除草剂容器', 37: '剩菜剩饭', 38: '大骨头', 39: '果壳瓜皮', 40: '残枝落叶', 41: '水果果皮', 42: '水果果肉', 43: '茶叶渣', 44: '菜梗菜叶', 45: '落叶', 46: '蛋壳', 47: '西餐糕点', 48: '鱼骨', 49: '一次性餐具', 50: '化妆品瓶', 51: '卫生纸', 52: '尿片', 53: '污损塑料', 54: '烟蒂', 55: '牙签', 56: '破碎花盆及碟碗', 57: '竹筷', 58: '纸杯', 59: '贝壳'}
def get_4class(a):  # 由60分类索引得4分类索引
    if a >= 0 and a < 29:
        return 0
    elif a < 37:
        return 1
    elif a < 49:
        return 2
    elif a < 60:
        return 3
    else:
        return -1

def getLabel(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        data = {'一级分类': ['图片格式有误', '图片格式有误', '图片格式有误'],
                '二级分类': ['图片格式有误', '图片格式有误', '图片格式有误'],
                '相似度': [0, 0, 0]}
        return jsonify(data)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = transform(img)
    img = img.view(1, -1, 224, 224).to(device)

    net.eval()
    data = {}
    output = []
    with torch.no_grad():
        output = net(img.float())
        _, pred = torch.max(output.data, 1)
        data['一级分类'] = labelIndex_4[get_4class(pred.item())]
        output = output.data.cpu().numpy()[0]
        outputs = softmax(output)
        output = outputs.tolist()

    output = np.around(output, 2)  # 保留两位小数
    np.set_printoptions(suppress=True)
    output = list(zip(output, range(len(output))))
    output = sorted(output, reverse=True)
    print(output)
    data['一级分类'] = [labelIndex_4[get_4class(output[0][1])], labelIndex_4[get_4class(output[1][1])],
                    labelIndex_4[get_4class(output[2][1])]]
    data['二级分类'] = [labelIndex_60[output[0][1]], labelIndex_60[output[1][1]], labelIndex_60[output[2][1]]]
    data['相似度'] = [output[0][0], output[1][0], output[2][0]]
    return data


@app.route('/getlabel', methods=['get', 'post'])
def getlabel():
    path = f'./static/images/image_{count}.jpg'
    img = request.files['imageFile']    # 从前端传来的图像
    img.save(path)

    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        data = {'一级分类': ['图片格式有误', '图片格式有误', '图片格式有误'],
                '二级分类': ['图片格式有误', '图片格式有误', '图片格式有误'],
                '相似度': [0, 0, 0]}
        return jsonify(data)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = transform(img)
    img = img.view(1, -1, 224, 224).to(device)

    net.eval()
    data = {}
    output = []
    with torch.no_grad():
        output = net(img.float())
        _, pred = torch.max(output.data, 1)
        data['一级分类'] = labelIndex_4[get_4class(pred.item())]
        output = output.data.cpu().numpy()[0]
        outputs = softmax(output)
        output = outputs.tolist()

    output = np.around(output, 2)   # 保留两位小数
    np.set_printoptions(suppress=True)
    output = list(zip(output, range(len(output))))
    output = sorted(output, reverse=True)
    # print(output)
    data['一级分类'] = [labelIndex_4[get_4class(output[0][1])], labelIndex_4[get_4class(output[1][1])], labelIndex_4[get_4class(output[2][1])]]
    data['二级分类'] = [labelIndex_60[output[0][1]], labelIndex_60[output[1][1]], labelIndex_60[output[2][1]]]
    data['相似度'] = [output[0][0], output[1][0], output[2][0]]
    return jsonify(data)


@app.route('/')
def test():
    return render_template('imagetest.html')


# 表单提交路径，需要指定接受方式
@app.route('/imagetest', methods=['GET', 'POST'])
def imagetest():
    # 通过表单中name值获取图片
    img = request.files['file']
    # 保存图片
    path = f'./static/images/image_{count}.jpg'
    img.save(path)
    data = getLabel(path)
    return jsonify(data)

if __name__ == '__main__':
    app.run()
