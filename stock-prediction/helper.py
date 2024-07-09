import numpy as np
def create_dataset(df):
    x=[]
    y=[]
    for i in range(50,df.shape[0]):
        x.append(df[i-50:i,0])
        y.append(df[i,0])
    x=np.array(x)
    y = np.array(y)
    return x,y
