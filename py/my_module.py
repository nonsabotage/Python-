import chardet
from collections import Counter
import copy
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.tokenfilter import ExtractAttributeFilter
from janome.tokenfilter import POSStopFilter
from janome.tokenfilter import POSKeepFilter
import re
import pathlib
from pprint import pprint
from gensim import (corpora, models)
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import japanize_matplotlib
import matplotlib.font_manager as fm
from matplotlib import rcParams
    
def get_string_from_file(filename):
    with open(filename, "rb") as f:
        d = f.read()
        e = chardet.detect(d)["encoding"]
        if e is None:
            e = "UTF-8"
        return d.decode(e)


def get_snippet_from_file(fielname, query, width = 2):
    s = get_string_from_file(fielname)
    p = '.{0,%d}%s.{0,%d}' % (width, query, width)
    r = re.search(p, s)
    if r :
        return r.group(0)
    else : 
        return None

def get_m_snippet_from_file(filename, query, width = 3):
    t = Tokenizer(wakati = True)
    qlist = list(t.tokenize(query))
    qlen  = len(qlist)
    s = get_string_from_file(filename)
    slist = list(t.tokenize(s))

    for i in [k for k, v in enumerate(slist) if v == qlist[0]]:
        if qlist == slist[i:i + qlen]:
            return "".join(slist[max(0, i-width):i + width + qlen])
    return None


def get_ngram(string, N=1):
    return [string[i:(i+N)] for i in range(len(string) - N + 1)]

def get_most_common_ngram(filename, N = 1, k = 1):
    s = get_string_from_file(filename)
    return Counter(get_ngram(s, N)).most_common(k)


def create_word_cloud(frequencies, font, width=600, height=400):
    wordcloud = WordCloud(background_color = "white", font_path = font, width=width, height=height)
    plt.figure(figsize = (width/50, height/50))
    plt.imshow(wordcloud.generate_from_frequencies(frequencies))
    plt.axis("off")
    plt.show()

def get_words(s, keep_pos = None):
    filters = []
    if keep_pos is None:
        filters.append(POSStopFilter(["記号"]))
    else:
        filters.append(POSKeepFilter(keep_pos))
    filters.append(ExtractAttributeFilter("surface"))
    a = Analyzer(token_filters = filters)
    return list(a.analyze(s))

def get_words_from_file(f):
    return get_words(get_string_from_file(f))



japanese_font_candidates = ['Hiragino Maru Gothic Pro', 'Yu Gothic',
                            'Arial Unicode MS', 'Meirio', 'Takao',
                            'IPAexGothic', 'IPAPGothic', 'VL PGothic',
                            'Noto Sans CJK JP']

def get_japanese_fonts(candidates=japanese_font_candidates):
    fonts = []
    for f in fm.findSystemFonts():
        p = fm.FontProperties(fname=f)
        try:
            n = p.get_name()
            if n in candidates:
                fonts.append(f)
        except RuntimeError:
            pass
    # サンプルデータアーカイブに含まれているIPAexフォントを追加
    fonts.append('font/ipaexg.ttf')
    return fonts

def configure_fonts_for_japanese(fonts=japanese_font_candidates):
    if hasattr(fm.fontManager, 'addfont'):
        fm.fontManager.addfont('irpb-files/font/ipaexg.ttf')
    else:
        ipa_font_files = fm.findSystemFonts(fontpaths='font')
        ipa_font_list = fm.createFontList(ipa_font_files)
        fm.fontManager.ttflist.extend(ipa_font_list)
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = fonts

def plot_frequecy(count, log_scale = False):
    y = list(sorted(count.values(), reverse = True))
    x = range(1, len(y) + 1)
    if log_scale:
        plt.loglog(x, y)
    else:
        plt.plot(x, y)


def build_corpus(file_list, dic_file = None, corpus_file = None):
    docs = []
    for f in file_list:
        text  = get_string_from_file(f)
        words = get_words(text, keep_pos = ["名詞"])
        docs.append(words)
        print(f)
    
    dic = corpora.Dictionary(docs)
    if not (dic_file is None):
        dic.save(dic_file)
    bows = [dic.doc2bow(d) for d in docs]
    if not (corpus_file is None):
        corpora.MmCorpus.seialize(corpus_file, bows)
    return dic, bows


def bows_to_cfs(bows):
    cfs = dict()
    for b in bows:
        for id, f in b:
            if not id in cfs:
                cfs[id] = 0
            cfs[id] += int(f)
    return cfs


def load_dictionary_and_corpus(dic_file, corpus_file):
    dic  = corpora.Dictionary.load(dic_file)
    bows = list(corpora.MmCorpus(corpus_file))
    if not hasattr(dic, "cfs"):
        dic.cfs = bows_to_cfs(bows)
    return dic, bows

def load_aozora_corpus():
    return load_dictionary_and_corpus("irpb-files/data/aozora/aozora.dic", "irpb-files/data/aozora/aozora.mm")

def get_bows(texts, dic, allow_update = False):
    bows = []
    for text in texts:
        words = get_words(text, keep_pos=["名詞"])
        bow   = dic.doc2bow(words, allow_update=allow_update)
        bows.append(bow)
    return bows

import copy


def add_to_corpus(texts, dic, bows, replicate = False):
    if replicate:
        dic  = copy.copy(dic)
        bows = copy.copy(bows)
    texts_bows = get_bows(texts, dic, allow_update = True)
    bows.extend(texts_bows)
    return dic, bows, texts_bows


def get_weights(bows, dic, tfidf_model, surface = False, N = 1000):
    weights = tfidf_model[bows]
    weights = [sorted(w, key = lambda x:x[1], reverse = True)[:N] for w in weights]
    if surface:
        return [[(dic[x], y) for x, y, *_ in w] for w in weights]
    else:
        return weights




def translate_bows(bows, table):
    return [[tuple([table[j[0]], j[1]]) for j in i if j[0] in table] for i in bows]

def get_tfidfmodel_and_weights(texts, use_aozora=True, pos=['名詞']):
    if use_aozora:
        dic, bows = load_aozora_corpus()
    else:
        dic = corpora.Dictionary()
        bows = []
    
    text_docs = [get_words(text, keep_pos=pos) for text in texts]
    text_bows = [dic.doc2bow(d, allow_update=True) for d in text_docs]
    bows.extend(text_bows)
    
    # textsに現れる語のidとtoken(表層形)のリストを作成
    text_ids = list(set([text_bows[i][j][0] for i in range(len(text_bows)) for j in range(len(text_bows[i]))]))
    text_tokens = [dic[i] for i in text_ids]
    
    # text_bowsにない語を削除．
    dic.filter_tokens(good_ids=text_ids)
    # 削除前後のIDの対応づけ
    # Y = id2id[X] として古いid X から新しいid Y が得られるようになる
    id2id = dict()
    for i in range(len(text_ids)):
        id2id[text_ids[i]] = dic.token2id[text_tokens[i]]
    
    # 語のIDが振り直されたのにあわせてbowを変換
    bows = translate_bows(bows, id2id)
    text_bows = translate_bows(text_bows, id2id)
    
    # TF・IDFモデルを作成
    tfidf_model = models.TfidfModel(bows, normalize=True)
    # モデルに基づいて重みを計算
    text_weights = get_weights(text_bows, dic, tfidf_model)
    
    return tfidf_model, dic, text_weights





# list 4.1
def jaccard(X, Y):
    x = set(X)
    y = set(Y)
    a = len(x.intersection(y))
    b = len(x.union(y))
    if b == 0:
        return 0
    else:
        return a / b


#  Listing 4.4 #

from gensim.similarities import MatrixSimilarity

def vsm_search(texts, query):
    tfidf_model, dic, text_weights = get_tfidfmodel_and_weights(texts)

    index = MatrixSimilarity(text_weights,  num_features=len(dic))

    # queryのbag-of-wordsを作成し，重みを計算
    query_bows = get_bows([query], dic)
    query_weights = get_weights(query_bows, dic, tfidf_model)

    # 類似度計算
    sims = index[query_weights[0]]

    # 類似度で降順にソート
    return sorted(enumerate(sims), key=lambda x: x[1], reverse=True)

def get_list_from_file(file_name):
    with open(file_name, 'r', encoding='UTF-8') as f:
        return f.read().split()
