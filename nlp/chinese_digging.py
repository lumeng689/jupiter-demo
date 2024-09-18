import jieba

jieba.suggest_freq('沙瑞金', True)
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('京州', True)

txt1 = '沙瑞金赞叹易学习的胸怀，是金山的百姓有福，可是这件事对李达康的触动很大。易学习又回忆起他们三人分开的前一晚，大家一起喝酒话别，易学习被降职到道口县当县长，王大路下海经商，李达康连连赔礼道歉，觉得对不起大家，他最对不起的是王大路，就和易学习一起给王大路凑了5万块钱，王大路自己东挪西撮了5万块，开始下海经商。没想到后来王大路竟然做得风生水起。沙瑞金觉得他们三人，在困难时期还能以沫相助，很不容易。'
doc_cut1 = jieba.cut(txt1)
result1 = ' '.join(doc_cut1)

print(result1)

txt2 = '沙瑞金向毛娅打听他们家在京州的别墅，毛娅笑着说，王大路事业有成之后，要给欧阳菁和她公司的股权，她们没有要，王大路就在京州帝豪园买了三套别墅，可是李达康和易学习都不要，这些房子都在王大路的名下，欧阳菁好像去住过，毛娅不想去，她觉得房子太大很浪费，自己家住得就很踏实。'
doc_cut2 = jieba.cut(txt2)
result2 = ' '.join(doc_cut2)

from sklearn.feature_extraction.text import TfidfVectorizer

#从文件导入停用词表
stpwrdpath = "stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
#将停用词表转换为list
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()

corpus = [result1, result2]
vector = TfidfVectorizer(stop_words=stpwrdlst)
tfidf = vector.fit_transform(corpus)
print(tfidf)

wordlist = vector.get_feature_names_out()#获取词袋模型中的所有词
# tf-idf矩阵 元素a[i][j]表示j词在i类文本中的tf-idf权重
weightlist = tfidf.toarray()
#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
for i in range(len(weightlist)):
    print("-------第",i,"段文本的词语tf-idf权重------")
    for j in range(len(wordlist)):
        print(wordlist[j],weightlist[i][j])

