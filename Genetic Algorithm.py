import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics
import random

#Function we will minimise
def f(x, y):
    return x*np.sin(4*x+1.1*y*np.sin(2*y))

x= np.linspace(0,10, 50)
y= np.linspace(0,10, 50)

mutation =0.2
popsize=20

xchrom=np.random.choice(x, size=popsize)
xchrom=xchrom.tolist()
ychrom=np.random.choice(y, size=popsize)
ychrom=ychrom.tolist()
fitt=list()

for i in range (popsize):
    fit=f(xchrom[i], ychrom[i])
    fitt.append(fit)

    hold = pd.DataFrame(list(zip(xchrom, ychrom, fitt)), columns = ['x','y','fitness'])
    #print(hold)
n=1
while len(hold)> 1:

    #Selection
    selection= statistics.median(fitt)
    hold=hold.loc[hold['fitness'] < selection]
    #print(hold)

    newgen = pd.DataFrame(columns=list(["x", 'y', "fitness" ]))

    for i in range(popsize-1):
    #Crossing over and evolution 
        b=0.5
        par=hold.sample(2)
        par.reset_index(drop=True)
        #print(par)
        xd=par.iloc[0].at['x']
        xm=par.iloc[1].at['x']
        xn=(1-b)*xm+b*xd

        yd=par.iloc[0].at['y']
        ym=par.iloc[1].at['y']
        yn=(1-b)*ym+b*yd

        new=[xn, yn, f(xn,yn)]

        #print(new)

        newgen.loc[i]=new
        
        i+=1

    #print(newgen)

    hold= newgen

    print(hold)
    mutx=random.choice(xchrom)
    muty=random.choice(ychrom)
    popsize+=-1

    #print(popsize)
    n+=1