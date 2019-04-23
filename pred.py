import numpy as np

with np.load('lottos_1-855_(855,45).npz') as a :
    lottos = a['lottos']
neg_lottos = 1-lottos
for x in range(1,neg_lottos.shape[0]):
    neg_lottos[x] = neg_lottos[x]*(neg_lottos[x]+neg_lottos[x-1])

neg_lottos = neg_lottos.astype(np.int16)
tomap = np.zeros((45,50,50))
for x in range(45):
    for y in range(neg_lottos.shape[0]-1):
        f,t  = neg_lottos[y,x], neg_lottos[y+1, x]
        tomap[x,f,t]+=1
#    print(x,'\n',tomap[x,:,0])
#for x in range(45):
#    print(x,'\n', tomap[x,neg_lottos[-1,x],:])
##########################################
pred = np.zeros(45)
for times in range(5):
    for x in range(45):
        pred[x] += tomap[x,neg_lottos[-times,x],times]
#print(pred)
##########################################
import matplotlib.pyplot as plt
%matplotlib inline
print(lottos.shape, neg_lottos.shape)
for times in range(850,neg_lottos.shape[0]-1):
    pred = np.zeros(45)
    for x in range(45):
        pred[x] += tomap[x,neg_lottos[times,x],0]
    #print(pred, neg_lottos[times])
    p1 = plt.bar(np.arange(45),pred)
    p2 = plt.bar(np.arange(45),lottos[times+1], bottom=pred)
    plt.axhline(np.mean(pred), color="orange")
    plt.axhline(np.median(pred), color="red")
    plt.show()
########################################
times=-1
pred = np.zeros(45)
for x in range(45):
    pred[x] += tomap[x,neg_lottos[times,x],0]
#print(pred, neg_lottos[times])
p1 = plt.bar(np.arange(45),pred)
#p2 = plt.bar(np.arange(45),lottos[times+1], bottom=pred)
plt.axhline(np.mean(pred), color="orange")
plt.axhline(np.median(pred), color="red")
plt.show()

########################################
print(np.argwhere(pred>=np.median(pred)).reshape(-1)+1)

print(np.argwhere(pred>np.median(pred)).reshape(-1)+1)

a = np.argwhere(pred>np.median(pred)).reshape(-1)

pp = pred/np.sum(pred)
print(pp, '\n')
print(np.sort(np.random.choice(45, 6, replace=False, p=pp)+1))
print(np.sort(np.random.choice(45, 6, replace=False, p=pp)+1))
print(np.sort(np.random.choice(45, 6, replace=False, p=pp)+1))
print(np.sort(np.random.choice(45, 6, replace=False, p=pp)+1))
print(np.sort(np.random.choice(45, 6, replace=False, p=pp)+1))

################################################################################
################################################################################

correct = np.where(lottos[-1]>0.5)[0]+1
pred = np.zeros(45)
pred_per = np.zeros(45)
for x in range(45):
    pred[x] = tomap[x,neg_lottos[-1,x],0]
    pred_per[x] = tomap[x,neg_lottos[-1,x],0]/np.sum(tomap[x,neg_lottos[-1,x]])

print(correct)
print(neg_lottos[-1])
print(pred)
print( pred_per)

################################################################################
################################################################################


result = np.zeros((4,7), dtype=np.int32)
ppred = pred/np.sum(pred)
ppred_per = pred_per / np.sum(pred_per)
ppp = ppred*ppred_per
ppp = ppp/np.sum(ppp)

for i in range(100):
    rand = np.random.choice(45, 6, replace=False)+1
    count = np.intersect1d(correct,rand).shape[0]
    result[0,count]+=1

    rand = np.random.choice(45, 6, replace=False, p=ppred)+1
    count = np.intersect1d(correct,rand).shape[0]
    result[1,count]+=1

    rand = np.random.choice(45, 6, replace=False, p=ppred_per)+1
    count = np.intersect1d(correct,rand).shape[0]
    result[2,count]+=1

    rand = np.random.choice(45, 6, replace=False, p=ppp)+1
    count = np.intersect1d(correct,rand).shape[0]
    result[3,count]+=1
    
    if i%10000 == 9999:
        print("# ", end='')
print()
print(result)

################################################################################
################################################################################


rand = np.random.choice(45, 6, replace=False, p=ppp)+1
print(np.sort(rand))
rand = np.random.choice(45, 6, replace=False, p=ppp)+1
print(np.sort(rand))
rand = np.random.choice(45, 6, replace=False, p=ppp)+1
print(np.sort(rand))
rand = np.random.choice(45, 6, replace=False, p=ppp)+1
print(np.sort(rand))
rand = np.random.choice(45, 6, replace=False, p=ppp)+1
print(np.sort(rand))
