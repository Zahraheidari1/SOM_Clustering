from preprocessing import np
import requests as rq
def fetch(sub,hon,size,dir):
    ids=np.array([])
    docs=np.array([])
    real_labels=np.array([])
    auth=rq.auth.HTTPBasicAuth('e2HjtDWCBnFf6NoqnmVtRg','SZnj5HHidWOEKpk7F8W4XFMN0Q51Lw')
    data={'grant_type':'password','username':'Former-Image1721','password':'armin1379'}
    headers={'User-Agent':'MyAPI/0.0.1'}
    res=rq.post('https://www.reddit.com/api/v1/access_token',auth=auth,data=data,headers=headers)
    T=res.json()['access_token']
    headers['Authorization']=f'bearer {T}'
    data=list()
    label=0
    for s in sub.split(','):
        res=rq.get(f'https://oauth.reddit.com/r/{s}/{hon}',headers=headers,params={'limit':f'{size}'})
        data.append(res.json()['data']['children'])
    for i in data:
        for post in i:
            with open(f'{dir}\{post["kind"]+post["data"]["id"]}.txt','w',encoding='utf-8') as f:
                f.write(post["data"]["title"]+'\n'+post['data']['selftext'])
            docs=np.append(docs,post["data"]["title"]+'\n'+post['data']['selftext'])
            ids=np.append(ids,post["kind"]+post["data"]["id"])
            real_labels=np.append(real_labels,label)
        label+=1
    np.save(dir+'\\real_labels.npy',real_labels)
    return (ids,docs)