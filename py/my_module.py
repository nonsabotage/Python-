import chardet
from collections import Counter
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.tokenfilter import ExtractAttributeFilter
from janome.tokenfilter import POSStopFilter
import re
import pathlib
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

def get_words(s):
    a = Analyzer(token_filters = [POSStopFilter(["記号"]), ExtractAttributeFilter("surface")])
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