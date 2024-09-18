# 安装pip install pyenchant
from enchant.checker import SpellChecker

chkr = SpellChecker("en_US")
chkr.set_text("Many peope likee to watch In the Name of People.")

for err in chkr:
    print("Error: ", err.word)

from nltk.stem import SnowballStemmer

# 提取词干
stemmer = SnowballStemmer("english")  # Choose a language
w1 = stemmer.stem("countries")  # Stem a word
print(w1)

# 词型还原
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
print(wnl.lemmatize('countries'))