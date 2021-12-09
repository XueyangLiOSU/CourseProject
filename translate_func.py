from textblob import TextBlob

def correct_misspel(sentence):
    '''
    input: 'whatp are ypu doing for this momment?'
    output: 'what are you doing for this moment?'
    '''
    cor_sen = TextBlob(sentence)
    return cor_sen.correct()

def detect_lang(sentence):
    '''
    input: '你在干什么'
    output: 'zh-CN'
    '''
    dect_senc = TextBlob(sentence)
    return dect_senc.detect_language()

def translate_2en(sentence):
    '''
    input:'你在干什么?'
    output: 'What are you doing?'
    '''
    tran_sen = TextBlob(sentence)
    return tran_sen.translate(to = 'en')