import nltk

# nltk.download('punkt')
# nltk.download('maxent_ne_chunker')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('words')
# nltk.download('treebank')

# 下载全部   python -m nltk.downloader all

sentence = """At eight o'clock on Thursday morning
... Arthur didn't feel very good."""

tokens = nltk.word_tokenize(sentence)

print(tokens)

tagged = nltk.pos_tag(tokens)
print(tagged[0:6])

entities = nltk.chunk.ne_chunk(tagged)
print(entities)

from nltk.corpus import treebank
t = treebank.parsed_sents('wsj_0001.mrg')[0]
t.draw()