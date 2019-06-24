import numpy as np
import matplotlib.pyplot as plt

def plot(isTrain):
    #True for plotting training result, else for validation res
    name='train'if isTrain else 'val'
    loss= np.loadtxt('model/loss_%s.txt'%name)

    x=loss[:,0]
    inds=[i for i in range(len(x))]
    # inds=[i for i in range(len(x)) if abs(int(x[i]*2)/2-x[i])<0.01]#用来增减图上选点的密集程度

    x=x[inds]
    loss_all=loss[inds,1]
    loss_sc=loss[inds,2]
    loss_loc=loss[inds,3]

    plt.figure()
    plt.plot(x,loss_all,label='loss_all',color='red')
    plt.title('%s_loss_all'%name)
    plt.ylabel('loss_all')
    plt.xlabel('epoch')
    plt.savefig('model/%s_loss_all.png'%name, dpi=150)
    plt.clf()

    plt.figure()
    plt.plot(x,loss_sc,label='loss_score',color='green')
    plt.title('%s_loss_score'%name)
    plt.ylabel('loss_sc')
    plt.xlabel('epoch')
    plt.savefig('model/%s_loss_score.png'%name, dpi=150)
    plt.clf()

    plt.figure()
    plt.plot(x,loss_loc,label='loss_location',color='blue')
    plt.title('%s_loss_location'%name)
    plt.ylabel('loss_loc')
    plt.xlabel('epoch')
    plt.savefig('model/%s_loss_location.png'%name, dpi=150)
    plt.clf()

    plt.figure()
    plt.plot(x,loss_all,label='loss_all',color='red')
    plt.plot(x,loss_sc,label='loss_score',color='green')
    plt.plot(x,loss_loc,label='loss_location',color='blue')
    plt.xlabel('loss')
    plt.ylabel('epoch')
    plt.title("%s_loss"%name)
    plt.legend()
    plt.savefig('model/%s_loss.png'%name, dpi=150)
    plt.clf()

if __name__ == '__main__':
    plot(True)
    plot(False)
