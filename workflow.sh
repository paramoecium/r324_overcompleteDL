WORKING_DIR=/tmp2/yuchentsai/dim_reduction_via_compressed_sensing
OUTPUT_DIR=$WORKING_DIR/output_14_ksvd
INPUT_DATA=$WORKING_DIR/data/merge-2014-11-02_to_2014-11-16.dat
INPUT_LABEL=$WORKING_DIR/data/truth_data-2014-11-02_to_2014-11-16.txt
START=2

if [ $START -lt 2 ];then
	python $WORKING_DIR/window.py $INPUT_DATA $INPUT_LABEL $OUTPUT_DIR
elif [ $START -lt 3 ];then
	mkdir -p $OUTPUT_DIR
	python $WORKING_DIR/reduce.py $WORKING_DIR/output_14/windowed $INPUT_LABEL $OUTPUT_DIR/reduced 1200
elif [ $START -lt 4 ];then
	python $WORKING_DIR/svm_bracketing.py $OUTPUT_DIR/reduced/svm_total 1200
else
	echo "done"
fi
