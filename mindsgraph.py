import json
import nltk
from tqdm import tqdm
from langdetect import detect
from collections import defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objects as go

analyzer = SentimentIntensityAnalyzer()
f = open('result.json')
words = set(nltk.corpus.words.words())
data = json.load(f)
newdic = defaultdict(list)


for i in tqdm(data['messages']):
    sent = ""
    if type(i['text']) is list:
        for u in i:
            if type(u) is str:
                sent += u
    else:
        sent = i['text']
    if 'shib' in sent.lower() or 'doge' in sent.lower():
        if detect(sent.lower()) == 'en':
            try:
                newdic[i['date'][:10]].append(sent)
            except KeyError:
                newdic[i['date'][:10]] = sent
countday = {}
sentimentday = {}
for i, j in tqdm(newdic.items()):
    vs = 0
    countday[i] = len(newdic[i])
    for k in j:
        print(analyzer.polarity_scores(k))
        vs += analyzer.polarity_scores(k)['compound']
    sentimentday[i] = vs / countday[i]

fig = go.Figure([go.Bar(x=list(sentimentday.keys()), y=list(sentimentday.values()))])
fig.show()

fig1 = go.Figure([go.Bar(x=list(countday.keys()), y=list(countday.values()))])
fig1.show()

f.close()
