CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=12395 bash examples/hunyuanvideo/run_unified_sanity_check.sh 1 2 > test/test_data/tp1cp2_layer36.log &
pid1=$!
CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=12335 bash examples/hunyuanvideo/run_unified_sanity_check.sh 2 1 > test/test_data/tp2cp1_layer36.log &
pid2=$!


wait $pid1 
echo "finish tp1 cp2 $pid1"
wait $pid2
echo "finish tp2 cp1 $pid2"


