from github_utils import get_github_repo, upload_github_issue

import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(formatter={'int':lambda x : f'{x: 3}'})

import urllib.request as httpcl
import json
from datetime import date
import os
from joblib import Parallel, delayed, cpu_count
cpucnt = cpu_count()
startdate = date(2002,12,7)
enddate = date.today()
dday = (enddate - startdate).days

result_md_table = ""

result_md_table += f'{cpucnt}, {dday}, {dday//7)}\n'
lasttime = dday//7 +2

starttime = 1
starttime = lasttime-12

issue_title = f'{lasttime} 회차 '

games = 5

crawlNo=[]
nokeys=[f'drwtNo{x+1}' for x in range(6)]

multipool = Parallel(n_jobs=(cpucnt*4))

def get_lotto(no):
    r= json.load(httpcl.urlopen(f"https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={no}"))
    if r['returnValue'] != 'success':
        return None,None
    crawl =  [r[x] for x in r if x in nokeys]
    crawl.sort()
    rstr = f"{r['drwNo']} {r['drwNoDate']} {crawl}"
    return rstr,crawl

poolfn = delayed(get_lotto)

crawlNo = multipool(poolfn(x) for x in range(starttime, lasttime) )

rstrs =  [ x for x,a in crawlNo if x != None]

result_md_table += f'{"\n".join( rstrs[-8:] )} \n'

crawlNo = [ x for a,x in crawlNo if x != None]


test=np.array(crawlNo)
#print(test.shape)
test = np.sort(test,axis=1)
#print(test[-8:,:])

recent_8_count = np.bincount(test[-8:,:].ravel(),minlength = 45)

#print(recent_8_count)

prob = np.ones(46)
prob[recent_8_count>=3]=0
prob[recent_8_count==2]=0.5

prob[0] = 0
prob = prob / np.sum(prob)

#print(prob)

final = np.stack([np.random.choice(46, 6, replace=False, p = prob) for x in range(30000)])
#print(np.sort(final[0])+1)

final = np.sort(final, axis=1)
# final = np.sort(final, axis=0)
final_uniq, count = np.unique(final, axis=0, return_counts=True)
max_count_idx = np.argsort(-count)
beautify_print_str = [ list("-"*45) for _ in range(games)]
final_selected = final_uniq[max_count_idx[:games]]
final_selected_count = count[max_count_idx[:games]]
for game_no in range(games):
    result_md_table += f'{final_selected[game_no], final_selected_count[game_no]}\n'
    for no in final_selected[game_no]:
        beautify_print_str[game_no][no-1] = "#"
    beautify_print_str[game_no] = "".join(beautify_print_str[game_no])

#print(beautify_print_str)
resultstr=[]
for x in range(0,45,7):
    tt = " | ".join([f'{temp[x:x+7]:7}' for temp in beautify_print_str])
    resultstr.append(tt)
result_md_table += f'{"\n".join(resultstr)}\n'
print(result_md_table)
access_token = os.environ['MY_GITHUB_TOKEN']
repository_name = "lotto"
    
repo = get_github_repo(access_token, repository_name)
upload_github_issue(repo, issue_title, result_md_table)
    
