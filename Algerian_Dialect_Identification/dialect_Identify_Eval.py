# -*- coding: utf-8 -*-
"""
Created on Sun May 13 07:36:10 2018

@author: DELL
"""
from sklearn.externals import joblib


# Define ner_test
def dli_eval(phrase, modelName, model, vect, embed):
    y_pred=[]
    x=[phrase]
    test_str = vect.transform(x)
    test_tfstr = embed.transform(test_str)
    test_tfstr.shape
    
    # model
    y_pred=model.predict(test_tfstr.toarray())[0]
    #print("DLI of The Phrase %s: "%phrase)
    # Representing Result of lsvm
    #print("\nBy %s is:"%modelName,y_pred)
    return y_pred



if __name__ == '__main__':
    import sys
    f = open("./test/results/Eval_msa_alg.out", 'w',encoding='utf-8')
    sys.stdout = f
    # Load Models
    # Lsvm
    fileLsvm = './model/dli_lsvm.pkl'
    lsvm=joblib.load(fileLsvm)
    # Bnb
    fileBnb = './model/dli_bnb.pkl'
    bnb=joblib.load(fileBnb)
    # Mnb
    fileMnb = './model/dli_mnb.pkl'
    mnb=joblib.load(fileMnb)
    #Load Data to Evaluate
    vectorizer = joblib.load("./tools/vectorizer.pkl")
    tfidf = joblib.load("./tools/tfidf.pkl")
    data=open('./test/Alg.txt','r',encoding='utf-8').read().split('\n')
    phrase=[d for d in data]
	  
    #Save dli_test for lsvm model
    clsvm=0
    with open("./test/results/eval_dli_lsvm.txt","w", encoding="utf-8") as file:
        file.write("phrase-------class-----------\n")
        for s in phrase:
            file.write(s)
            file.write('\t\t')
            out=dli_eval(s,'LSVM', lsvm, vectorizer, tfidf)
            file.write(str(out))
            file.write('***\t\t***')
            file.write('\t\t')
            if out=="ALG":
                clsvm=clsvm+1
            file.write('\n')
        print('lsvm Accuracy for Alg is: ',clsvm/len(phrase))   
    #Save dli_test for bnb model
    clbnb=0
    with open("./test/results/eval_dli_bnb.txt","w", encoding="utf-8") as file:
        file.write("phrase-------class-----------\n")
        for s in phrase:
            file.write(s)
            file.write('\t\t')
            out=dli_eval(s,'BNB', bnb, vectorizer, tfidf)
            file.write(str(out))
            file.write('***\t\t***')
            file.write('\t\t')
            if out=="ALG":
                clbnb=clbnb+1
            file.write('\n')
        print('bnb Accuracy for Alg is: ',clbnb/len(phrase))       
    #Save dli_test for mnb model
    clmnb=0
    with open("./test/results/eval_dli_mnb.txt","w", encoding="utf-8") as file:
        file.write("phrase-------class-----------\n")
        for s in phrase:
            file.write(s)
            file.write('\t\t')
            out=dli_eval(s,'MNB', mnb, vectorizer, tfidf)
            file.write(str(out))
            file.write('***\t\t***')
            file.write('\t\t')
            if out=="ALG":
                clmnb=clmnb+1
            file.write('\n')
        print('mnb Accuracy for Alg is: ',clmnb/len(phrase)) 	 
    
    f.close()