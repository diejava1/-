from flask import Flask, request,jsonify,session,flash,Response
from flask import render_template,redirect,url_for
from control import *
import json




app = Flask("my-app")
import flask

from MyTest2 import *


classes=['finance','realty','stocks','education','science','politics','sports','game','entertainment']
vote={}
f=open('D:\py\pycharmProject\demo\HelloWorld\THUCNews\data\class.txt','r',encoding='utf-8')
content=f.readlines()
for i in range(len(content)):
    vote[str(i)]=0
f.close()



data=""

@app.route('/zhuYe', methods=['GET','POST'])
def zhuYe():
    if request.method=='POST':
        return redirect('/aaa')
    return render_template('zhuYe.html')

@app.route('/')
def hello_world():
    return redirect(url_for('erWeiMa'))

@app.route('/erWeiMa')
def erWeiMa():
    return render_template('erWeiMa.html')

@app.route('/aaa', methods=['GET','POST'])
def aaa():
    global data
    if request.method=='POST':
        for key in vote.keys():
            vote[key]=0


        models=['TextCNN','TextRNN','TextRCNN','TextRNN_Att','DPCNN','FastText','Transformer']
        data = (request.form.get('message'))
        #chosed=[False,False,False,False,False,False,False]
        xuan=(request.form.get('TextCNN'))

        chosed={}
        for i in range(len(models)):
            chosed[models[i]]=False

        for i in range(len(models)):
            w=request.form.get(models[i])
            if w=='1':
                chosed[models[i]]=True

        print(chosed)
        print(data)
        f=open('D:\py\pycharmProject\demo\HelloWorld\THUCNews\data\MyTest3.txt','w',encoding='utf-8')
        xie=data+'\t'+'9'+'\n'
        for i in range(1000):
            f.write(xie)

        f.close()
        print(chosed)

        for key,value in chosed.items():
            if value:
                print('我在这!!!!!!!!!!!!!',key)
                ceshi(key)
                f=open('D:\py\pycharmProject\demo\HelloWorld\预测结果.txt','r',encoding='utf-8')
                result=f.read()
                vote[result]+=1
                f.close()

        print('!!!!!!!!!!!!!',vote)

        #f=open('D:\py\pycharmProject\demo\HelloWorld\static\json\dataV.json','w',encoding='utf-8')
        #f.close()
        #qianmian='{tooltip: {trigger: \'item\'},legend: {top: \'5%\',left: \'center\'},series: [{name: \'Access From\',type: \'pie\',radius: [\'40%\', \'70%\'],avoidLabelOverlap: false,itemStyle: {borderRadius: 10,borderColor: \'#fff\',borderWidth: 2},label: {show: false,position: \'center\'},emphasis: {label: {show: true,fontSize: \'40\',fontWeight: \'bold\'}},labelLine: {show: false},'






        f=open('D:\py\pycharmProject\demo\HelloWorld\预测结果.txt','w',encoding='utf-8')

        for key,value in vote.items():
            xie=str(key)+':'+str(value)+'\n'
            f.write(xie)
        f.close()

    return render_template('aaa.html')


@app.route('/dataVisualize')
def dataVisualize():


    return render_template('datavisualize.html')



@app.route('/dataV')
def dataV():
    dataList={}
    '''
    tooltip={}

    tooltip['trigger']='item'
    dataList['tooltip']=tooltip
    legend={}
    legend['top']='5%'
    legend['left']='center'
    dataList['legend']=legend

    series=[]
    series.append({})
    series[0]['name'] = 'Access Form'
    series[0]['type'] = 'pie'
    series[0]['radius'] = ['40%', '70%']
    series[0]['avoidLabelOverlap'] = False

    itemStyle={}
    itemStyle['borderRadius']=10
    itemStyle['borderColor'] = '#fff'
    itemStyle['borderWidth'] = 2
    series[0]['itemStyle']=itemStyle

    label={}
    label['show']=False
    label['position']='center'
    series[0]['label'] = label

    emphasis={}
    label={}
    label['show']=True
    label['fontSize']='40'
    label['fontWeight']='bold'
    emphasis['label']=label
    series[0]['emphasis']=emphasis

    labelLine={}
    labelLine['show']=False
    
'''

    tooltip = {}

    tooltip['trigger'] = 'item'
    dataList['tooltip'] = tooltip
    legend = {}
    legend['top'] = '5%'
    legend['left'] = 'center'
    dataList['legend'] = legend

    series=[
        {
            'name': 'Access From',
            'type': 'pie',
            'radius': ['40%', '70%'],
            'avoidLabelOverlap': False,
            'itemStyle': {
                'borderRadius': 10,
                'borderColor': '#fff',
                'borderWidth': 2
            },
            'label': {
                'show': False,
                'position': 'center'
            },
            'emphasis': {
                'label': {
                    'show': True,
                    'fontSize': '40',
                    'fontWeight': 'bold'
                }
            },
            'labelLine': {
                'show': False
            },
            'data': [
                {'value': 0, 'name': 'finance'},
                {'value': 0, 'name': 'realty'},
                {'value': 0, 'name': 'stocks'},
                {'value': 0, 'name': 'education'},
                {'value': 0, 'name': 'science'},
                {'value': 0 , 'name' : 'society'},
                {'value': 0, 'name': 'politics'},
                {'value': 0, 'name': 'sports'},
                {'value': 0, 'name': 'game'},
                {'value': 0, 'name': 'entertainment'}

            ]
        }
    ]

    print(series)
    print(series[0]['data'])
    print(series[0]['data'][0])

    print(vote)


    for key,value in vote.items():
        print(key,value)
        series[0]['data'][eval(key)]['value']=value


    dataList['series'] = series

    #for key,value in vote.items():
        #dataList[series][0]['data'][eval(key)]['value']=value

    #(dataList[series][0]['data'])



    return (json.dumps({'tooltip':tooltip,'legend':legend,'series':series}))













if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)





