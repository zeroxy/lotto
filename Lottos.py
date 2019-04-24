import numpy as np
import requests
import pyquery


class Lottos:
    def __init__(self, filename='lottos_1-855_(855,45).npz'):
        self.filename = filename
        try:
            with np.load(self.filename) as npz :
                self.lottos = npz['lottos']
            print("loaded!!")
        except Exception as e:
            self.lottos=np.zeros((0,45),dtype=np.int16)
            self.update()
        finally:
            self.neg_lottos = 1-self.lottos
            self.neg_lottos = self.neg_lottos.astype(np.int16)
            for x in range(1,self.neg_lottos.shape[0]):
                self.neg_lottos[x] = self.neg_lottos[x]*(self.neg_lottos[x]+self.neg_lottos[x-1])
            self.tomap = np.zeros((45,60,2),np.int32)
            for x in range(45):
                for y in range(self.neg_lottos.shape[0]-1):
                    f,t  = self.neg_lottos[y,x], self.neg_lottos[y+1, x]
                    if f>59:
                        print(x, y, f , t)
                    if t == 0:
                        self.tomap[x,f,0]+=1
                    else:
                        self.tomap[x,f,1]+=1
        
                    
    def __len__(self):
        return self.lottos.shape[0]
    
    def update(self):
        url = 'https://search.naver.com/search.naver?sm=tab_drt&where=nexearch&query=9999회로또'
        body = requests.get(url)
        d = pyquery.PyQuery(body.text)('._lotto-btn-current em')
        limit = int(d.html()[:-1])
        end = self.lottos.shape[0]
        if limit>end:
            self.lottos = np.append(self.lottos,np.zeros((limit-end,45)),axis=0)
            print(self.lottos.shape)
            for no in range(end,limit):
                url = f'https://search.naver.com/search.naver?sm=tab_drt&where=nexearch&query={no+1}회로또'
                body = requests.get(url)
                d = pyquery.PyQuery(body.text)('.num_box .num')
                idx = [ int(x.text)-1 for x in d]
                self.lottos[no,idx]=1
                print('\r'+str(no),end='')
            np.savez_compressed(self.filename, lottos=self.lottos)
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
        #print("###\n", predict_seed, np.cov(predict_seed), np.mean(predict_seed))
        for x in range(45):
            rate = 1.
            for y in range(history):
                percent = (self.tomap[x,predict_seed[y,x],0]/np.sum(self.tomap[x,predict_seed[y,x]]))
                probability[x] += percent *rate
                rate = rate * (1-percent)
        return (probability/np.sum(probability))
    
    def get_real_history(self, no):
        assert 1< no <= self.lottos.shape[0]
        return np.where(self.lottos[no-1]>0.5)[0]+1
    
    def recommend(self, prob=None, count=5):
        if prob is None:
            p = np.ones((45))/45
        result = np.zeros((count,6))
        for i in range(count):
            result[i] = np.sort(np.random.choice(45, 6, replace=False, p=prob)+1)
        result.astype(np.int8)
        return result
