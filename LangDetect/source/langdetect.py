import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from utils import *
from classifiers import *
from preprocess import  preprocess

seed = 42
random.seed(seed)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", 
                        help="Input data in csv format", type=str)
    parser.add_argument("-v", "--voc_size", 
                        help="Vocabulary size", type=int)
    parser.add_argument("-a", "--analyzer",
                         help="Tokenization level: {word, char}", 
                        type=str, choices=['word','char'])
    #add a boolean argument to the parser
    parser.add_argument('-o', "--output", action='store_true', default = False ,help='Want an output file?')
    parser.add_argument("-c", "--classifier",
                        help="Classifier to use: {knn, svc, nb, rf}",)
    parser.add_argument("--ngram", help="ngram range", nargs=2, default=(1,1), type=int)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    raw = pd.read_csv(args.input)
    
    # Languages
    languages = set(raw['language'])
    print('========')
    print('Languages', languages)
    print('========')

    # Split Train and Test sets
    X=raw['Text']
    y=raw['language']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    print('========')
    print('Split sizes:')
    print('Train:', len(X_train))
    print('Test:', len(X_test))
    print('========')
    
    # Preprocess text (Word granularity only)
    if args.analyzer == 'word':
        X_train, y_train = preprocess(X_train,y_train)
        X_test, y_test = preprocess(X_test,y_test)

    #Compute text features
    features, X_train_raw, X_test_raw = compute_features(X_train, 
                                                            X_test, 
                                                            analyzer=args.analyzer, 
                                                            max_features=args.voc_size,
                                                            ngram_range=args.ngram)
    if args.output:
        output_dir = '../output/' + args.classifier + "_" + args.analyzer + "_" + str(args.voc_size) + "_" + str(args.ngram)



    print('========')
    print('Number of tokens in the vocabulary:', len(features))
    print('Coverage: ', compute_coverage(features, X_test.values, 
                                         analyzer=args.analyzer, output_file=output_dir + '.txt' if args.output else None))
    print('========')


    #Apply Classifier  
    X_train, X_test = normalizeData(X_train_raw, X_test_raw)
    if args.classifier == 'knn':
        y_predict = applyKNN(X_train, y_train, X_test)
    elif args.classifier == 'svc':
        y_predict = applySVC(X_train, y_train, X_test)
    elif args.classifier == 'rf':
        y_predict = applyRandomForest(X_train, y_train, X_test)
    else:
        y_predict = applyNaiveBayes(X_train, y_train, X_test)
    
    print('========')
    print('Prediction Results:')    
    plot_F_Scores(y_test, y_predict, output_dir + '.txt' if args.output else None)
    print('========')
    
    plot_Confusion_Matrix(y_test, y_predict, "Greens", output_dir + '_conf_matrix.png' if args.output else None) 


    #Plot PCA
    print('========')
    print('PCA and Explained Variance:') 
    plotPCA(X_train, X_test,y_test, languages, output_dir + '_pca.png' if args.output else None) 
    print('========')
