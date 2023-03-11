# MUD-project
![Thank you image](https://www.rosette.com/wp-content/uploads/2014/11/thankyou.jpg)
This project contains a text classifier that can be used to classify text into the corresponding language .
It uses different classifiers, such as k-nearest neighbors (KNN), support vector machine (SVM), random forest, and naive Bayes to perform the classification.

## Prerequisites

Before running the program, make sure you have the following installed:

-   Python 3
-   pandas
-   scikit-learn
-   matplotlib
-   numpy

## Usage

The program can be executed using the following command:

phpCopy code

`python text_classifier.py -i <input_file> -v <vocabulary_size> -a <analyzer> -c <classifier> --ngram <ngram_range>` 

where:

-   `<input_file>`: The path to the input CSV file containing text data.
-   `<vocabulary_size>`: The number of words or characters to be used in the vocabulary.
-   `<analyzer>`: The tokenization level. Can be either 'word' or 'char'.
-   `<classifier>`: The classifier to be used. Can be one of 'knn', 'svc', 'nb', or 'rf'.
-   `<ngram_range>`: The range of n-grams to be used. Should be a tuple of two integers.

Additionally, you can use the following optional argument:

-   `-o`: If specified, an output file will be generated containing the prediction results.

## Example

`python text_classifier.py -i input.csv -v 5000 -a word -c knn --ngram 1 3 -o` 

This will classify the text in `input.csv` using a KNN classifier with a vocabulary size of 5000 and word-level tokenization. It will use n-grams of 1 to 3 words and generate an output file containing the prediction results.

## Output

The program generates the following output:

-   The languages present in the input file.
-   The sizes of the train and test sets.
-   The number of tokens in the vocabulary and its coverage of the test data.
-   The prediction results in the form of an F-score plot and a confusion matrix.
-   The PCA plot and the explained variance of the data.

If the `-o` option is specified, the prediction results will be written to a file named `<classifier>_<analyzer>_<vocabulary_size>_<ngram_range>.txt` in the `output` directory. The confusion matrix and PCA plot will be saved in the same directory with names `<classifier>_<analyzer>_<vocabulary_size>_<ngram_range>_conf_matrix.png` and `<classifier>_<analyzer>_<vocabulary_size>_<ngram_range>_pca.png`, respectively.

## Launching multiple experiments

The following script can be used to launch multiple experiments with different configurations:

```console
for i in knn svc nb rf
do
    for j in 1000 2000 5000
    do
        for k in "1 1" "1 2" "1 3"
        do
            echo "python langdetect.py -i ../data/dataset.csv -v $j -a word -c $i -o --ngram $k"
            python langdetect.py -i ../data/dataset.csv -v $j -a word -c $i -o --ngram $k
            echo "python langdetect.py -i ../data/dataset.csv -v $j -a char -c $i -o --ngram $k"
            python langdetect.py -i ../data/dataset.csv -v $j -a char -c $i -o --ngram $k
        done
    done
done
```

This script launches the `langdetect.py` script with different configurations for the `vocabulary-size`, `analyzer`, `classifier`, and `ngram-range` parameters. It generates output files for each experiment if the `-o` flag is set.


## `experimentsToPandas.py`

This script reads all text files in the output directory and extracts information from their filenames and contents to create a pandas dataframe. Each text file should contain information on the performance of a classifier in the language detection task. 
The resulting pandas dataframe will have the following columns:

-   `Classifier`: The name of the classifier used.
-   `Kind`: The kind of input used (either "char" or "word").
-   `Vocabulary Size`: The size of the vocabulary used.
-   `Ngram 1`: The first integer specifying the n-gram range used.
-   `Ngram 2`: The second integer specifying the n-gram range used.
-   `Coverage`: The coverage score for the classifier on the language detection task.
-   `F1 (micro)`: The micro-averaged F1 score for the classifier on the language detection task.
-   `F1 (macro)`: The macro-averaged F1 score for the classifier on the language detection task.
-   `F1 (weighted)`: The weighted F1 score for the classifier on the language detection task.

The resulting dataframe is sorted by `F1 (weighted)` and `Coverage` in descending order, and is saved to a csv file at `../output/results.csv`.
