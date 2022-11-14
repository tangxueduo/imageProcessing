@staticmethod
def Rotate(img:BMPFile,angle:float,islinear:bool,extend:bool):
  tempimg = deepcopy(img)
  data = tempimg.data
  angle = radians(angle)
  wd0=tempimg.bmInfo.biWidth
  ht0=abs(tempimg.bmInfo.biHeight)
  #计算新图的尺寸
  if extend:
    wd = int(np.round(abs(cos(angle))*wd0+abs(sin(angle)*ht0)))
    ht = int(np.round(abs(sin(angle))*wd0+abs(cos(angle)*ht0)))
  else:
    wd=wd0;ht=ht0
  #旋转后的坐标系
  i1 = np.arange(0,wd,1)-wd//2
  j1 = np.arange(0,ht,1)-ht//2
  i1,j1 = np.meshgrid(i1,j1)
  #获取旋转前原图的坐标
  i = i1 * cos(angle) + j1 * sin(angle)
  j = -i1 * sin(angle) + j1 * cos(angle)
  #位置修正
  i = i + wd0//2 + 1
  j = j + ht0//2 + 1
  #加黑边
  Extdata = np.zeros((ht0+2,wd0+2,3))
  Extdata[1:ht0+1,1:wd0+1] = data
  data = Extdata
  #处理
  if(not islinear):
    tempimg.data = TranRotate.Nearly(data,i,j)
  else:
    tempimg.data = TranRotate.Linear(data,i,j)
  return tempimg 
  
#临近法
@staticmethod
def Nearly(data:np.ndarray,i:np.ndarray,j:np.ndarray):
  ht0,wd0 = data.shape[0],data.shape[1]
  ht,wd = i.shape
  i = np.floor(i).astype(np.int32)
  j = np.floor(j).astype(np.int32)
  i = np.clip(i,0,wd0-1)
  j = np.clip(j,0,ht0-1)
  return data[j,i].astype(np.uint8)
  
#插值法
@staticmethod
def Linear(data:np.ndarray,i:np.ndarray,j:np.ndarray):
  ht0,wd0 = data.shape[0],data.shape[1]
  ht,wd = i.shape
  #坐标取整，左小右大，上小下大，通过这四个参数的组合得出四个采样点的坐标
  iL = np.floor(i).astype(np.int32)
  iR = iL+1
  jU = np.floor(j).astype(np.int32)
  jD = jU+1
  #利用取整的坐标计算四个角的权重，注意RGB数组需要复制三份以向量化运算
  pUL= (jD-j)*(iR-i);pUL=np.repeat(np.expand_dims(pUL,2),3,2)
  pUR= (jD-j)*(i-iL);pUR=np.repeat(np.expand_dims(pUR,2),3,2)
  pDL= (j-jU)*(iR-i);pDL=np.repeat(np.expand_dims(pDL,2),3,2)
  pDR= (j-jU)*(i-iL);pDR=np.repeat(np.expand_dims(pDR,2),3,2)
  #钳制点的范围
  iL = np.clip(iL,0,wd0-1);iR = np.clip(iR,0,wd0-1)
  jU = np.clip(jU,0,ht0-1);jD = np.clip(jD,0,ht0-1)
  #权重与变换后的点相乘
  data = data[jU,iL]*pUL+data[jU,iR]*pUR+data[jD,iL]*pDL+data[jD,iR]*pDR
  return data.astype(np.uint8)
