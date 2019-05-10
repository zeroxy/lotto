from flask import Flask, jsonify, render_template
from Lottos import Lottos
import numpy as np
import json
import os
import datetime as dayt
from datetime import datetime as dt

app = Flask(__name__)
KST = dayt.timezone(dayt.timedelta(hours=9))
first_date = dt(2002,12,7,20,50,tzinfo=KST)

@app.route('/')
def get_the_luck():
    def aug_pb(pb, pb_pow=3):
        pb=pb**pb_pow
        pb=pb/np.sum(pb)
        return pb

    lottos = Lottos()
    gamedaydelta = dt.now(tz=KST)-first_date
    gametimes = gamedaydelta.days//7+1
    pb, seed = lottos.get_probability()
    r= lottos.recommend(pb, 10)
    result_obj={}
    result_obj = {
        'recent_correct':lottos.get_real_history(len(lottos)).tolist(),
	'recent_time' : len(lottos),
	'calc_time' : gametimes
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
    pb = aug_pb(pb,3)
    r= lottos.recommend(pb, 10)
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
    pb = aug_pb(pb,4)
    r= lottos.recommend(pb, 10)
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
