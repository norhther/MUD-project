import nltk
import pandas as pd


#Tokenizer function. You can add here different preprocesses.
def preprocess(sentence, labels):
    '''
    Task: Given a sentence apply all the required preprocessing steps
    to compute train our classifier, such as sentence splitting, 
    tokenization or sentence splitting.

    Input: Sentence in string format
    Output: Preprocessed sentence either as a list or a string
    '''
    # Place your code here
    # Keep in mind that sentence splitting affectes the number of sentences
    # and therefore, you should replicate labels to match.

    #sentence splitting
    sentence = sentence.apply(nltk.sent_tokenize)
    # tokenization
    sentence = sentence.apply(lambda x: [nltk.word_tokenize(item) for item in x])
    
    labels_splitted = []
    sentences_splitted= []
    for i, value in enumerate(sentence.values):
        for sent_value in value:
            current_sentence = []
            for word in sent_value:
                current_sentence.append(word)
            sentences_splitted.append(current_sentence)
            labels_splitted.append(labels.iloc[i])

    sentence = pd.Series(sentences_splitted)
    labels = pd.Series(labels_splitted)
    # sentence = sentence.apply(lambda x: [item.lower() for item in x])
    # sentence = sentence.apply(lambda x: [item for item in x if item.isalpha()])
    #sentence = sentence.apply(lambda x: [item for item in x if len(item) > 2]) removing words with 2 characters or less
    # lemmatization
    #lemmatizer = nltk.stem.WordNetLemmatizer()
    #sentence = sentence.apply(lambda x: [lemmatizer.lemmatize(item) for item in x])
    # stemming
    stemmer = nltk.stem.PorterStemmer()
    sentence = sentence.apply(lambda x: [stemmer.stem(item) for item in x])
    
    sentence = sentence.apply(lambda x: ' '.join(x))


    return sentence,labels



