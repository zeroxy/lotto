from flask import Flask, jsonify, render_template
from Lottos import Lottos
import numpy as np
import json
import os

app = Flask(__name__)

@app.route('/')
def get_the_luck():
    def aug_pb(pb, pb_pow=3):
        pb=pb**pb_pow
        pb=pb/np.sum(pb)
        #pb = pb + (1/100)
        #pb = pb / np.sum(pb)
        return pb

    def correct_test(i, pb_power=1, testtime = 70000):
        starttime = dt.now()
        pb = lottos.get_probability(i)
        pb = aug_pb(pb, pb_pow=pb_power)
        y_ = lottos.get_real_history(i)
        count_pb = np.bincount(np.sum(np.isin( lottos.recommend(prob=pb, count=testtime),y_), axis=1), minlength=7)
        count_no = np.bincount(np.sum(np.isin( lottos.recommend(count=testtime),y_)         , axis=1), minlength=7)
        #(np.sort(np.argsort(-pb)[(y_-1)])+1)}\
        ''''''
        print(f"\n========  {i} : {(y_)} ========\
        \npred : {count_pb}\nunif : {count_no}\nrate : {(count_pb/count_no)[:-1]}\
        \npredict_acc : {count_pb/testtime}\npred_cumsum : {np.cumsum(count_pb/testtime)}\
        \nuniform_acc : {count_no/testtime}\nunif_cumsum : {np.cumsum(count_no/testtime)}\
        \n{dt.now()-starttime}\n")
        
        pb_correct_rate = np.sum(count_pb[3:])/np.sum(count_pb[:3])
        no_correct_rate = np.sum(count_no[3:])/np.sum(count_no[:3])
        return (count_pb / testtime)


    lottos = Lottos()

    pb, seed = lottos.get_probability()
    r= lottos.recommend(pb)
    result_obj={}
    result_obj = {
        'recent_correct':lottos.get_real_history(len(lottos)).tolist()
    }
    result_obj['uniform'] = {
            'recommend':r.astype(np.int16).tolist(),
            'seed':seed.tolist(),
            'pb':pb.tolist()
    }
    result_txt = ""

    result_txt +=f'{lottos.get_real_history(len(lottos))}\n'
    result_txt +=f'{r}\n'
    result_txt +=f'{seed}\n'
    result_txt +=f'{pb}\n'
    result_txt +=f'{lottos.validation_history(r)}\n'

    result_txt +="\n\n===================\n\n"

    pb, seed = lottos.get_probability()
    r= lottos.recommend(pb)
    pb = aug_pb(pb)
    r= lottos.recommend(pb)
    result_obj['triple'] = {
            'recommend':r.astype(np.int16).tolist(),
            'seed':seed.tolist(),
            'pb':pb.tolist()
    }
    result_txt +=f'{r}\n'
    result_txt +=f'{pb}\n'
    result_txt +=f'{lottos.validation_history(r)}\n'


    result_txt +="\n\n===================\n\n"

    pb, seed = lottos.get_probability()
    r= lottos.recommend(pb)
    pb = aug_pb(pb,4)
    r= lottos.recommend(pb)
    result_obj['quad'] = {
            'recommend':r.astype(np.int16).tolist(),
            'seed':seed.tolist(),
            'pb':pb.tolist()
    }
    result_txt +=f'{r}\n'
    result_txt +=f'{pb}\n'
    result_txt +=f'{lottos.validation_history(r)}\n'
    print(result_txt)
    #return jsonify({'result':result_obj})
    return render_template('index.html',result=result_obj)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)