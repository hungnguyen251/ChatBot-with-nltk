import nltk
import warnings
 
warnings.filterwarnings("ignore")
# nltk.download() # for downloading packages
import tensorflow as tf
import numpy as np
import random
import string  # to process standard python strings
 
f = open('nlp python answer.txt', 'r', errors='ignore')
m = open('modules pythons.txt', 'r', errors='ignore')
checkpoint = "./chatbot_weights.ckpt"
# session = tf.compat.v1.InteractiveSession()
# session.run(tf.compat.v1.global_variables_initializer())
# saver = tf.train.Saver()
# saver.restore(session, checkpoint)
 
raw = f.read()
rawone = m.read()
raw = raw.lower()  # converts to lowercase
rawone = rawone.lower()  # converts to lowercase
# nltk.download('punkt') # first-time use only
# nltk.download('wordnet') # first-time use only
sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # converts to list of words

sent_tokensone = nltk.sent_tokenize(rawone)  # converts to list of sentences
word_tokensone = nltk.word_tokenize(rawone)  # converts to list of words
 
sent_tokens[:2]
sent_tokensone[:2]
 
word_tokens[:5]
word_tokensone[:5]
 
lemmer = nltk.stem.WordNetLemmatizer()
 
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
 
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
 
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
 
Introduce_Ans = ["My name is ALICE.", "My name is ALICE you can called me pi.", "Im ALICE :) ",
                 "My name is ALICE. and my nickname is pi and i am happy to solve your queries :) "]
GREETING_INPUTS = ("hello", "hi", "hiii", "hii", "hiiii", "hiiii", "xin chao", "xin chào", "chào bạn")
GREETING_RESPONSES = ["Chào bạn", "Xin chào, tôi có thể giúp gì cho bạn?", "Chào bạn, mình có thể giúp gì cho bạn?"]
Basic_Q = ("mình cần tư vấn về nước hoa?", "mình cần tư vấn về son moi?", "mình cần tư vấn về quần áo?")
Basic_Ans = ("Bạn đã từng sử dụng sản phâm nào rồi?")

Basic_Om = ("bạn có thể tư vấn cho tôi không?", "ban co the tu van cho toi khong?", "tu van cho toi?", "tư vấn cho tôi?", "tư vấn")
Basic_AnsM = ("Chắc chắn rồi, bạn hiện tại đang quan tâm về mặt hàng nào nhỉ?")

Basic_Qn = ("nước hoa.", "nuoc hoa.", "mình muốn tư vấn về nước hoa.")
Basic_AnsN = ("Nước hoa bên mình có rất nhiều loại bạn có thể thử với thương hiệu Chanel.")

Basic_Ql = ("son môi", "son", "son moi", "son li", "mình muốn tư vấn về son.")
Basic_AnsL = ("Bạn có thể thử loại son của Hàn Quốc rất được ưa chuộng hiện nay.")

Basic_Qp = ("bên bạn có chương trình khuyến mãi nào không", "bên bạn có chương trình km nào không", "bên bạn có chương trình khuyến mãi nào ko")
Basic_AnsP = ("Hiên tại shop đang triển khai chương trình khuyến mãi như sau: Khuyến mãi mua 1 tặng 1 trả tiền 2.")

Basic_Qt = ("giá cả thế nào vậy?", "gia bao nhieu vay?", "gia chai nay bao nhieu")
Basic_AnsT = ("Dạ hiện nay bên mình đang bán chai nước hoa của Chanel giá 3.000.000 nhé bạn.")

Basic_Qu = ("gia thoi nay bao nhieu", "gia thoi son nay bao nhieu")
Basic_AnsU = ("Dạ hiện nay bên mình đang bán thỏi 3CE này với giá 300.000 nhé bạn.")

Basic_Qv = ("ok cam on", "ok minh cam on", "cam on", "cảm ơn")
Basic_AnsV = ("Không có gì, bạn muốn tư vấn về sản phẩm cứ liên hệ bên mình.")
 
# Checking for greetings
def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
 
# Checking for Basic_Q
def basic(sentence):
    for word in Basic_Q:
        if sentence.lower() == word:
            return Basic_Ans
 
# Checking for Basic_QM
def basicM(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in Basic_Om:
        if sentence.lower() == word:
            return Basic_AnsM
        
# Checking for Basic_QN
def basicN(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in Basic_Qn:
        if sentence.lower() == word:
            return Basic_AnsN

# Checking for Basic_QL
def basicL(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in Basic_Ql:
        if sentence.lower() == word:
            return Basic_AnsL
        
# Checking for Basic_QP
def basicP(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in Basic_Qp:
        if sentence.lower() == word:
            return Basic_AnsP
        
# Checking for Basic_QP
def basicT(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in Basic_Qt:
        if sentence.lower() == word:
            return Basic_AnsT
        
# Checking for Basic_QP
def basicU(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in Basic_Qu:
        if sentence.lower() == word:
            return Basic_AnsU
        
# Checking for Basic_QP
def basicV(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in Basic_Qv:
        if sentence.lower() == word:
            return Basic_AnsV
 
# Checking for Introduce
def IntroduceMe(sentence):
    
    return random.choice(Introduce_Ans)
 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# Generating response
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response
 
# Generating response
def responseone(user_response):
    robo_response = ''
    sent_tokensone.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokensone)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokensone[idx]
        return robo_response
 
def chat(user_response):
    user_response = user_response.lower()
    keyword = " module "
    keywordone = " module"
    keywordsecond = "module"
 
    if user_response != 'bye':
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False
            # print("ROBO: You are welcome..")
            return "You are welcome.."
        elif basicM(user_response) is not None:
            return basicM(user_response)
        elif basicN(user_response) is not None:
            return basicN(user_response)
        elif basicL(user_response) is not None:
            return basicL(user_response)
        elif basicP(user_response) is not None:
            return basicP(user_response)
        elif basicT(user_response) is not None:
            return basicT(user_response)
        elif basicU(user_response) is not None:
            return basicU(user_response)        
        elif basicV(user_response) is not None:
            return basicV(user_response)                
        else:
            if (user_response.find(keyword) != -1 or user_response.find(keywordone) != -1 or user_response.find(
                    keywordsecond) != -1):
                # print("ROBO: ",end="")
                # print(responseone(user_response))
                return responseone(user_response)
                sent_tokensone.remove(user_response)
            elif greeting(user_response) is not None:
                # print("ROBO: "+greeting(user_response))
                return greeting(user_response)
            elif (user_response.find("your name") != -1 or user_response.find(" your name") != -1 or user_response.find(
                    "your name ") != -1 or user_response.find(" your name ") != -1):
                return IntroduceMe(user_response)
            elif basic(user_response) is not None:
               return basic(user_response)
            elif basicN(user_response) is not None:
               return basicN(user_response)
           
            else:
                # print("ROBO: ",end="")
                # print(response(user_response))
                return response(user_response)
                sent_tokens.remove(user_response)
 
    else:
        flag = False
        # print("ROBO: Bye! take care..")
        return "Bye! take care.."

            