import ast, pickle
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MOVIES = 'tmdb_5000_movies.csv'
CREDITS = 'credits.csv.gz'
CACHE = 'model.pkl'

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

ps = PorterStemmer()
cv = CountVectorizer(max_features=5000, stop_words='english')

def parse(obj, key=None, limit=None):
    try: arr = ast.literal_eval(obj)
    except: return []
    out = []
    for d in arr:
        if key and d.get('job')!=key: continue
        out.append(d.get('name'))
        if limit and len(out)>=limit: break
    return out

def stem(t): return ' '.join(ps.stem(w) for w in str(t).split())

def build():
    import os, pickle
    if os.path.exists(CACHE): return pickle.load(open(CACHE,'rb'))
    m = pd.read_csv(MOVIES); c = pd.read_csv(CREDITS, compression='gzip')
    df = m.merge(c,on='title')[['movie_id','title','overview','genres','keywords','cast','crew']].dropna()
    df['genres']=df.genres.apply(lambda x:[i['name'] for i in ast.literal_eval(x)])
    df['keywords']=df.keywords.apply(lambda x:[i['name'] for i in ast.literal_eval(x)])
    df['cast']=df.cast.apply(lambda x:parse(x,limit=3))
    df['crew']=df.crew.apply(lambda x:parse(x,key='Director'))
    df['tags']= (df.overview+df.genres+df.keywords+df.cast+df.crew).astype(str)
    df['tags']=df['tags'].apply(stem)
    vec=cv.fit_transform(df['tags']).toarray()
    sim=cosine_similarity(vec)
    data={'df':df.reset_index(drop=True),'sim':sim}
    pickle.dump(data,open(CACHE,'wb')); return data

DATA=build()

@app.get('/recommend')
def recommend(title: str=Query(...), k:int=5):
    df, sim = DATA['df'], DATA['sim']
    try: idx=df.index[df.title.str.lower()==title.lower()][0]
    except: raise HTTPException(404,'Title not found')
    scores=sorted(list(enumerate(sim[idx])), key=lambda x:x[1], reverse=True)[1:k+1]
    return {'input':df.iloc[idx].title,'recommendations':[{'title':df.iloc[i].title,'score':float(s)} for i,s in scores]}

if __name__=='__main__':
    import uvicorn; uvicorn.run(app,host='0.0.0.0',port=8000)
