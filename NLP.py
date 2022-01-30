import pandas as pd
import kss
from krwordrank.hangle import normalize
from krwordrank.word import KRWordRank
from krwordrank.sentence import summarize_with_sentences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')


etext = 'Time is an illusion. Lunchtime double so!'
ktext='딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어려워요. 농담아니에요. 이제 해보면 알걸요?'
print(etext.split())##通过指定分隔符对字符串进行切片
print(ktext.split())
from nltk.tokenize import word_tokenize
print(word_tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example = "Family is not an important thing. It's everything."
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(example)
ss = "is not"
stop_words = word_tokenize(ss)
result = []
for w in word_tokens:
    if w not in stop_words:
        result.append(w)

docs = [
  'basically as soon as the rape jokes start, you start degrading the gay community and talking about how anyone that s suicidal is a monster',
  'Make no mistake, theyll ridicule u, 2! @rickygee15 Most of ugirls should go out with no make up at all tonight.. No one would recognize you'
]
set([1,2,3,4])##创建一个无序和无重复元素的集合

frist.s = docs[1].split()