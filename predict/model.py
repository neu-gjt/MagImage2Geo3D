from tkinter import *
from tkinter import filedialog
from keras import models
import numpy as np
from PIL import Image,ImageTk
import os
import pynoddy
import pynoddy.history
import copy
import pynoddy.output
from imp import reload
import math
import scipy.misc
from skimage import io

imgpath=''
fold='0'
fault='0'
tilt='0'
a=''
fold_pre=[[0,0]]
fault_pre=[[0,0]]
tilt_pre=[[0,0]]
fault_fold_pre=[[0,0,0,0]]
fault_tilt_pre=[[0,0,0,0]]
pre_label=[[0,0,0,0,0,0]]

    
def imgpredict():
    img=Image.open(imgpath)
    img = img.convert('RGB')
    global fold,fault,prediction,itype
    
    img_class = img.resize((75, 75),Image.ANTIALIAS)
    img_class=np.array(img_class)
    model = models.load_model(r'..\classification\inceptionv3.h5')
    prediction= model.predict(img_class.reshape(-1, 75, 75,3))
    if prediction[0][0]>0.5:
        itype=1
    elif prediction[0][1]>0.5:
        itype=2
    elif prediction[0][2]>0.5:
        itype=3
    elif prediction[0][3]>0.5:
        itype=4
    else:
        itype=5
    
    if itype==1:
        img = img.resize((40, 40),Image.ANTIALIAS)
        img=np.array(img)
        model = models.load_model(r'..\regression\fault.h5')
        fault_pre = model.predict(img.reshape(-1, 40, 40,3))
        pre_label[0][0]=0
        pre_label[0][1]=0
        pre_label[0][2]=fault_pre[0][0]
        pre_label[0][3]=fault_pre[0][1]
        pre_label[0][4]=0
        pre_label[0][5]=0
    elif itype==2:
        img = img.resize((40, 40),Image.ANTIALIAS)
        img=np.array(img)
        model = models.load_model(r'..\regression\fold.h5')
        fold_pre= model.predict(img.reshape(-1, 40, 40,3))
        pre_label[0][0]=fold_pre[0][0]
        pre_label[0][1]=fold_pre[0][1]
        pre_label[0][2]=0
        pre_label[0][3]=0
        pre_label[0][4]=0
        pre_label[0][5]=0
    elif itype==3:
        img = img.resize((40, 40),Image.ANTIALIAS)
        img=np.array(img)
        model = models.load_model(r'..\regression\tilt.h5')
        tilt_pre = model.predict(img.reshape(-1, 40, 40,3))
        pre_label[0][0]=0
        pre_label[0][1]=0
        pre_label[0][2]=0
        pre_label[0][3]=0
        pre_label[0][4]=tilt_pre[0][0]
        pre_label[0][5]=tilt_pre[0][1]
        
    elif itype==4:
        img = img.resize((40, 40),Image.ANTIALIAS)
        img=np.array(img)
        model = models.load_model(r'..\regression\fault_fold.h5')
        fault_fold_pre = model.predict(img.reshape(-1, 40, 40,3))
        pre_label[0][0]=fault_fold_pre[0][2]
        pre_label[0][1]=fault_fold_pre[0][3]
        pre_label[0][2]=fault_fold_pre[0][0]
        pre_label[0][3]=fault_fold_pre[0][1]
        pre_label[0][4]=0
        pre_label[0][5]=0
    elif itype==5:
        img = img.resize((40, 40),Image.ANTIALIAS)
        img=np.array(img)
        model = models.load_model(r'..\regression\tilt_fault.h5')
        tilt_fault_pre = model.predict(img.reshape(-1, 40, 40,3))
        pre_label[0][0]=0
        pre_label[0][1]=0
        pre_label[0][2]=tilt_fault_pre[0][0]
        pre_label[0][3]=tilt_fault_pre[0][1]
        pre_label[0][4]=tilt_fault_pre[0][2]
        pre_label[0][5]=tilt_fault_pre[0][3]
    if itype==1 or itype==2 or itype==3:
        i=0
        for num in pre_label[0]:
            num=int(num*355+0.5)
            # num=math.floor(num)/100
            pre_label[0][i]=num
            i+=1
    if itype==4 or itype==5:
        i=0
        for num in pre_label[0]:
            num=int(num*340+0.5)
            # num=math.floor(num)/100
            pre_label[0][i]=num
            i+=1
    global fold_DipDir_Entry,fold_Dip_Entry,fault_DipDir_Entry,fault_Dip_Entry,tilt_DipDir_Entry,tilt_Dip_Entry
    global fold_pro_Entry,fault_pro_Entry,tilt_pro_Entry,fold_fault_pro_DipDir_Entry,tilt_fault_pro_Entry
    fold_pro_Entry.delete(0, END)
    fault_pro_Entry.delete(0, END)
    tilt_pro_Entry.delete(0, END)
    fold_fault_pro_Entry.delete(0, END)
    tilt_fault_pro_Entry.delete(0, END)
    fold_pro_Entry.insert(0,"{:.5f}".format(prediction[0][1]))
    fault_pro_Entry.insert(0,"{:.5f}".format(prediction[0][0]))
    tilt_pro_Entry.insert(0,"{:.5f}".format(prediction[0][2]))
    fold_fault_pro_Entry.insert(0,"{:.5f}".format(prediction[0][3]))
    tilt_fault_pro_Entry.insert(0,"{:.5f}".format(prediction[0][4]))
    fold_DipDir_Entry.delete(0, END)
    fold_Dip_Entry.delete(0, END)
    fault_DipDir_Entry.delete(0, END)
    fault_Dip_Entry.delete(0, END)
    tilt_DipDir_Entry.delete(0, END)
    tilt_Dip_Entry.delete(0, END)
    fault_DipDir_Entry.insert(0,pre_label[0][2])
    fault_Dip_Entry.insert(0,pre_label[0][3])
    fold_DipDir_Entry.insert(0,pre_label[0][0])
    fold_Dip_Entry.insert(0,pre_label[0][1])
    tilt_DipDir_Entry.insert(0,pre_label[0][4])
    tilt_Dip_Entry.insert(0,pre_label[0][5])

def openfile():
    global imgpath,pathshow,imLabel1,img1,savepath_Label
    imgpath=filedialog.askopenfilename(title='Select magnetic field images', filetypes=[('JEPG', '*.JPG'), ('All Files', '*')])
    pathshow.delete(0, END)
    pathshow.insert(0,'.../'+os.path.basename(imgpath))
    im1=Image.open(imgpath)
    im1 = im1.resize((160,160),Image.ANTIALIAS)
    img1=ImageTk.PhotoImage(im1)
    imLabel1.configure(image = img1)
    root.update()

def Gen3D():
    global a,img3,imLabel3
    
    if itype==1:
        his = pynoddy.history.NoddyHistory(r".\noddy\fault.his")
        a='fault'
        his_changed = copy.deepcopy(his)
        his_changed.events[2].properties['Dip Direction']=pre_label[0][2]
        his_changed.events[2].properties['Dip']=pre_label[0][3]
        his_changed.write_history(os.path.join(r"../program",a+'.his'))
    elif itype==2:
        his = pynoddy.history.NoddyHistory(r".\noddy\fold.his")
        a='fold'
        his_changed = copy.deepcopy(his)
        his_changed.events[2].properties['Dip Direction']=pre_label[0][0]
        his_changed.events[2].properties['Dip']=pre_label[0][1]
        his_changed.write_history(os.path.join(r"../program",a+'.his'))
    elif itype==3:
        his = pynoddy.history.NoddyHistory(r".\noddy\tilt.his")
        a='tilt'
        his_changed = copy.deepcopy(his)
        his_changed.events[2].properties['Plunge Direction']=pre_label[0][4]
        his_changed.events[2].properties['Rotation']=pre_label[0][5]
        his_changed.write_history(os.path.join(r"../program",a+'.his'))
    elif itype==4:
        his = pynoddy.history.NoddyHistory(r".\noddy\fold_fault.his")
        a='fold_fault'
        his_changed = copy.deepcopy(his)
        his_changed.events[2].properties['Dip Direction']=pre_label[0][2]
        his_changed.events[2].properties['Dip']=pre_label[0][3]
        his_changed.events[3].properties['Dip Direction']=pre_label[0][0]
        his_changed.events[3].properties['Dip']=pre_label[0][1]
        his_changed.write_history(os.path.join(r"../program",a+'.his'))
    elif itype==5:
        his = pynoddy.history.NoddyHistory(r".\noddy\tilt_fault.his") 
        a='tilt_fault'        
        his_changed = copy.deepcopy(his)
        his_changed.events[2].properties['Dip Direction']=pre_label[0][2]
        his_changed.events[2].properties['Dip']=pre_label[0][3]
        his_changed.events[3].properties['Plunge Direction']=pre_label[0][4]
        his_changed.events[3].properties['Rotation']=pre_label[0][5]
        his_changed.write_history(os.path.join(r"../program",a+'.his'))

    if Var2.get()==1:
        sec='x'
    elif Var2.get()==2:
        sec='y'
    his = pynoddy.history.NoddyHistory(os.path.join(r"../program",a+'.his'))
    his.determine_model_stratigraphy()
    output="model"
    history_name="model.his"
    his.write_history(history_name)
    pynoddy.compute_model(history_name,output)
    reload(pynoddy.output)
    h_out = pynoddy.output.NoddyOutput(output)
    h_out.plot_section(sec,
                   layer_labels = his.model_stratigraphy,
                   colorbar_orientation = 'horizontal',
                   colorbar=False,
                   title = '',
#                   savefig=True, fig_filename = 'fold_thrust_NS_section.eps',
                   cmap = 'YlOrRd',
                   savefig=True)
    im3=Image.open(os.path.join(r'..\program',output+'_section_'+sec+'_pos_26.png'))
    im3 = im3.resize((320,160),Image.ANTIALIAS)
    img3=ImageTk.PhotoImage(im3)
    imLabel3.configure(image = img3)
    savepath_Label.configure(text='Save file to ../program/'+a+'.his')
    root.update()

def CalMag():
    path=os.path.join(r"..\program",a+'.his')
    pynoddy.compute_model(path,a, sim_type = 'GEOPHYSICS')
    reload(pynoddy.output)
    geophys = pynoddy.output.NoddyGeophysics(a)
    scipy.misc.imsave(os.path.join(r'..\program',a+'.jpg'), geophys.mag_data)
    global imLabel4,img4
    im4=Image.open(os.path.join(r'../program',a+'.jpg'))
    im4 = im4.resize((160,160),Image.ANTIALIAS)
    img4=ImageTk.PhotoImage(im4)
    imLabel4.configure(image = img4)
    root.update()
#主窗体
root=Tk()
root.title('3D Geological Structure Inversion ')
root.wm_geometry('661x550+500+100')
root.wm_resizable(False,False)

zimu=Label(root,text='   ')
zimu.grid(row=0,column=0,columnspan=2,padx=10,pady=5)
#选择文件
# frame1=LabelFrame(root,text='Prediction by CNN')
# frame1.grid(row=1,column=0,sticky=W+E)

#显示磁场图像
frame2=LabelFrame(root,text='Input image')
frame2.grid(row=1,column=0,padx=5,sticky=N+S)
im1=Image.open(r'bg.jpg')
im1 = im1.resize((160,160),Image.ANTIALIAS)
img1=ImageTk.PhotoImage(im1)
imLabel1=Label(frame2,image=img1)
imLabel1.grid(row=0,column=0,columnspan=2,padx=10,pady=5)

pathshow=Entry(frame2, width=27,font =("Times",8,'bold'))
pathshow.grid(row=1,column=0,padx=5,pady=10,sticky=N+S)
imgchoose=Button(frame2, text='...',font =("Times",8,'bold'),width=2,height=1, command=openfile)
imgchoose.grid( row=1,column=1,padx=5,pady=5)

#显示预测结果
frame3=LabelFrame(root,text='Predicted results by CNN')
frame3.grid(row=1,column=1,ipadx=0,sticky=N+S)

frame9=LabelFrame(frame3,text='Classification model')
frame9.grid(row=0,column=0,ipadx=5,sticky=E+W)
foldLabel=Label(frame9,text='Probability of belonging to each category')
foldLabel.grid(row=0,column=2,columnspan=4,padx=10,pady=0,sticky=NW)
#fold_DipDir
fold_pro_Label=Label(frame9,text='fold')
fold_pro_Label.grid(row=1,column=4,padx=10,pady=0,sticky=N+S+W)
fold_pro_Entry=Entry(frame9, width=10)
fold_pro_Entry.grid(row=1,column=5,padx=0,pady=5,sticky=N+S+W)
#fold_Dip
fault_pro_Label=Label(frame9,text='fault')
fault_pro_Label.grid(row=1,column=2,padx=10,pady=0,sticky=N+S+W)
fault_pro_Entry=Entry(frame9, width=10)
fault_pro_Entry.grid(row=1,column=3,padx=0,pady=5,sticky=N+S+W)

tilt_pro_Label=Label(frame9,text='tilt')
tilt_pro_Label.grid(row=1,column=6,padx=10,pady=0,sticky=N+S+W)
tilt_pro_Entry=Entry(frame9, width=10)
tilt_pro_Entry.grid(row=1,column=7,padx=0,pady=5,sticky=N+S+W)
#fault_DipDir
fold_fault_pro_Label=Label(frame9,text='fold_fault')
fold_fault_pro_Label.grid(row=2,column=2,padx=10,pady=0,sticky=N+S+W)
fold_fault_pro_Entry=Entry(frame9, width=10)
fold_fault_pro_Entry.grid(row=2,column=3,padx=0,pady=5,sticky=N+S+W)
#fault_Dip
tilt_fault_pro_Label=Label(frame9,text='tilt_fault')
tilt_fault_pro_Label.grid(row=2,column=4,padx=10,pady=0,sticky=N+S+W)
tilt_fault_pro_Entry=Entry(frame9, width=10)
tilt_fault_pro_Entry.grid(row=2,column=5,padx=0,pady=5,sticky=N+S+W)

btpredict=Button(frame9, text='Predict',font =("Times",8,'bold'),width=8,height=1, command=imgpredict)
btpredict.grid(row=2,column=6,columnspan=4,padx=10,pady=5)

frame8=LabelFrame(frame3,text='Regression model')
frame8.grid(row=1,column=0,ipadx=5)
#fold
foldLabel=Label(frame8,text='Parameters of fold')
foldLabel.grid(row=0,column=4,columnspan=2,padx=10,pady=0,sticky=NW)
#fold_DipDir
fold_DipDir_Label=Label(frame8,text='Dip Dir')
fold_DipDir_Label.grid(row=1,column=4,padx=10,pady=0,sticky=N+S+W)
fold_DipDir_Entry=Entry(frame8, width=8)
fold_DipDir_Entry.grid(row=1,column=5,padx=0,pady=5,sticky=N+S+W)
#fold_Dip
fold_Dip_Label=Label(frame8,text='Dip')
fold_Dip_Label.grid(row=2,column=4,padx=10,pady=0,sticky=N+S+W)
fold_Dip_Entry=Entry(frame8, width=8)
fold_Dip_Entry.grid(row=2,column=5,padx=0,pady=5,sticky=N+S+W)

#fault
faultLabel=Label(frame8,text='Parameters of fault')
faultLabel.grid(row=0,column=2,columnspan=2,padx=10,pady=0,sticky=NW)
#fault_DipDir
fault_DipDir_Label=Label(frame8,text='Dip Dir')
fault_DipDir_Label.grid(row=1,column=2,padx=10,pady=0,sticky=N+S+W)
fault_DipDir_Entry=Entry(frame8, width=8)
fault_DipDir_Entry.grid(row=1,column=3,padx=0,pady=5,sticky=N+S+W)
#fault_Dip
fault_Dip_Label=Label(frame8,text='Dip')
fault_Dip_Label.grid(row=2,column=2,padx=10,pady=0,sticky=N+S+W)
fault_Dip_Entry=Entry(frame8, width=8)
fault_Dip_Entry.grid(row=2,column=3,padx=0,pady=5,sticky=N+S+W)

#tilt
tiltLabel=Label(frame8,text='Parameters of tilt')
tiltLabel.grid(row=0,column=6,columnspan=2,padx=10,pady=0,sticky=NW)
#fault_DipDir
tilt_DipDir_Label=Label(frame8,text='Plunge Dir')
tilt_DipDir_Label.grid(row=1,column=6,padx=10,pady=0,sticky=N+S+W)
tilt_DipDir_Entry=Entry(frame8, width=8)
tilt_DipDir_Entry.grid(row=1,column=7,padx=0,pady=5,sticky=N+S+W)
#fault_Dip
tilt_Dip_Label=Label(frame8,text='Rotation')
tilt_Dip_Label.grid(row=2,column=6,padx=10,pady=0,sticky=N+S+W)
tilt_Dip_Entry=Entry(frame8, width=8)
tilt_Dip_Entry.grid(row=2,column=7,padx=0,pady=5,sticky=N+S+W)


frame4=LabelFrame(root,text='Forward modeling by Noddy from predicted structure and parameters')
frame4.grid(row=2,column=0,columnspan=2,pady=5)
#生成并显示三维地质模型
frame5=LabelFrame(frame4,text='Section of model')
frame5.grid(row=0,column=0)
im3=Image.open(r'bg.jpg')
im3 = im3.resize((320,160),Image.ANTIALIAS)
img3=ImageTk.PhotoImage(im3)
imLabel3=Label(frame5,image=img3)
imLabel3.grid(row=0,column=0,rowspan=2,padx=5,pady=5)
Var2= IntVar()
R1 = Radiobutton(frame5, text = "X",variable=Var2, value=1)
R2 = Radiobutton(frame5, text = "Y", variable=Var2, value=2)
R1.grid(row=0,column=1,padx=0)
R2.grid(row=1,column=1,padx=0)
savepath_Label=Label(frame5,text=' ')
savepath_Label.grid(row=2,column=0,padx=0,pady=0)
btgen=Button(frame5, text='Model',font =("Times",8,'bold'),width=8,height=1, command=Gen3D)
btgen.grid(row=2,column=1,padx=5,pady=5)

frame6=LabelFrame(frame4,text='Magnetic data in Noddy')
frame6.grid(row=0,column=1)
im4=Image.open(r'bg.jpg')
im4 = im4.resize((160,160),Image.ANTIALIAS)
img4=ImageTk.PhotoImage(im4)
imLabel4=Label(frame6,image=img4)
imLabel4.grid(row=0,column=0,padx=20,pady=5)
btgen=Button(frame6, text='Forward images',font =("Times",8,'bold'),width=14,height=1, command=CalMag)
btgen.grid(row=1,column=0,padx=10,pady=5)


root.mainloop()