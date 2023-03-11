# launch 
# create a list with knn, svc, nb, rf
# iterate throug it and launc python langdetect.py -i ../data/dataset.csv -v 1000 -a word -c knn -o
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