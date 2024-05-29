echo '####################'
echo '#  Grading Script  #'
echo '####################'
echo

echo "Running test file 1"
python3 ngrams.py train1.txt test1.txt > yourTrace1.txt
ds=$( diff yourTrace1.txt trace1.txt )
if [ -z "$ds" ]; then
    echo "Correct!"
else
    echo "There are some errors!"
    diff yourTrace1.txt trace1.txt
fi

echo ""

echo "Running test file 2"
python3 ngrams.py train2.txt test2.txt > yourTrace2.txt
ds=$( diff yourTrace2.txt trace2.txt )
if [ -z "$ds" ]; then
    echo "Correct!"
else
    echo "There are some errors!"
    diff yourTrace2.txt trace2.txt
fi
