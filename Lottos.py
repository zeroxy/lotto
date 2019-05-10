## numpy (1.16.3)
## requests (2.21.0)
## joblib (0.12.5)
## pyquery (1.4.0)

import numpy as np
import requests
import pyquery
from datetime import datetime as dt
from time import sleep
from joblib import Parallel, delayed, cpu_count
np.set_printoptions(precision=8, suppress=True, linewidth=120)
#np.set_printoptions()

class Lottos:
    def __init__(self, filename='./lottos_db.npz'):
        self.filename = filename
        try:
            with np.load(self.filename) as npz :
                self.lottos = npz['lottos']
                self.add_lottos=npz['add_lottos']
            print("loaded!! ", self.lottos.shape)
        except Exception as e:
            print(e)
            self.lottos=np.zeros((0,45),dtype=np.int16)
            self.add_lottos=np.zeros((0,45),dtype=np.int16)
        finally:
            #self.update()
            self.create_statistic()
            
    def create_statistic(self):
        self.neg_lottos = (1-(self.lottos)).astype(np.int16) +self.add_lottos.astype(np.int16)
        self.tomap = np.zeros((45,70,2))
        for i in range(1,self.neg_lottos.shape[0]):
            self.neg_lottos[i] += self.neg_lottos[i]*self.neg_lottos[i-1]
        for x in range(45):
            bincnt = np.bincount(self.neg_lottos[:,x], minlength=69)
            for y in range(len(bincnt)-1):
                self.tomap[x,y,0] = bincnt[y]-bincnt[y+1]
                self.tomap[x,y,1] = bincnt[y+1]
                    
    def __len__(self):
        return self.lottos.shape[0]
    
    def get_url(self, no):
        return f'https://search.naver.com/search.naver?query={no+1}회로또'
    
    def update(self):
        starttime = dt.now()
        url = self.get_url(9998)
        body = requests.get(url)
        while body.status_code != 200:
            print(f'\r{body.status_code} {dt.now()}', end='')
            sleep(5)
            body = requests.get(url)
        d = pyquery.PyQuery(body.text)('._lotto-btn-current em')
        limit = int(d.html()[:-1])
        end = self.lottos.shape[0]
        
        def get_balls(no):
            result = []
            url = self.get_url(no)
            body = requests.get(url)
            while True:
                d = pyquery.PyQuery(body.text)('.num_box .num')
                result = [ int(x.text)-1 for x in d]
                if len(result)==7:
                    break
                sleep(5)
                body = requests.get(url)
            print(f'\r    {no+1} ', end='')
            if no%180 == 179:
                print(f'  {dt.now()-starttime}')
            return result
        
        if limit>end:
            self.lottos = np.append(self.lottos,np.zeros((limit-end,45)),axis=0)
            self.add_lottos = np.append(self.add_lottos,np.zeros((limit-end,45)),axis=0)
            print(self.lottos.shape)
            verb=0
            crawled = Parallel(n_jobs=10, backend='threading', verbose=verb)(
                delayed(get_balls)(x) for x in range(end,limit)
            )
            print(f'  {dt.now()-starttime}')
            for rowno,row in enumerate(crawled):
                self.lottos[end+rowno,row]=1
                self.add_lottos[end+rowno,row[-1]] = 1
            np.savez_compressed(self.filename, lottos=self.lottos, add_lottos=self.add_lottos)
            print(f'\nwe had {end}. so update to {limit}. now we have {self.lottos.shape[0]} rows.')
        else:
            print('\nno update')
        
    def get_probability(self, no=None, history=1):
        if no is None:
            no=self.lottos.shape[0]+1
        assert 1< no <= self.lottos.shape[0]+1
        assert history < no
        probability = np.zeros(45)
        predict_seed = self.neg_lottos[no-1-history:no-1]
        prob_bias = (1-np.power((38/45), predict_seed[-1]))*0.03
        #print(f'predict_seed{no} : \n{predict_seed}\n###\n{prob_bias}\n\
        #cov : {np.cov(predict_seed)}, mean : {np.mean(predict_seed)}')
        for x in range(45):
            rate = 1.
            for y in range(history):
                percent = (self.tomap[x,predict_seed[y,x],0]/
                           np.sum(self.tomap[x,predict_seed[y,x]]))
                probability[x] += percent *rate
                rate = rate * (1-percent)
        #probability += prob_bias
        return (probability/np.sum(probability)), predict_seed
    
    def get_real_history(self, no):
        assert 1< no <= self.lottos.shape[0]
        return np.where((self.lottos[no-1]-self.add_lottos[no-1])>0.5)[0]+1
    
    def recommend(self, prob=None, count=5):
        if prob is None:
            p = np.ones((45))/45
        result = np.zeros((count*3,6))
        for i in range(count*3):
            result[i] = np.sort(np.random.choice(45, 6, replace=False, p=prob)+1)
        result.astype(np.int8)
        uniqresult = np.unique(result, axis=0)
        if uniqresult.shape[0]<count:
            idxs = np.sort(np.random.choice(uniqresult.shape[0], count))
            print(f"## warning!! unique rows:{uniqresult.shape[0]} < count:{count}")
        else :
            idxs = np.sort(np.random.choice(uniqresult.shape[0], count, replace=False))
        return uniqresult[idxs]
      
    def validation_history(self, recommends):
        assert recommends is not None
        assert recommends.ndim == 2
        assert recommends.shape[1] == 6
        rr = np.zeros((recommends.shape[0], 45))
        for i, r in enumerate(recommends):
            rr[i,r.astype(np.int32)-1] = 1
        
        rrr = np.expand_dims(rr,1)
        lll = np.expand_dims((self.lottos-self.add_lottos), 0)
        temp = np.sum((lll-rrr)**2, axis=2)
        for tt in temp:
            print(np.bincount(tt.astype(np.int32))[::2])
