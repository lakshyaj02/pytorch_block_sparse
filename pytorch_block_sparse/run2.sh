#!/bin/bash
VAR1="wikitext2 perplexity"
VAR2="ptb perplexity"
VAR3="c4 perplexity"
VAR4="total_gb_size"
VAR5="time"
groupsize=16

for thresh in 0.001 0.005 0.01 0.05 0.1;
do
        for thresh_block in 0.1 0.2 0.5 0.9;
        do
                dict_name="_test"
                title="$subsamples$dict_name"
                echo -e "3/8, 8/8, $groupsize, $thresh, $subsamples, $thresh_block" >> res2.txt
                python test_quantize.py facebook/opt-350m pajama --sparse True --wbits 3 --spbits 8 --perchannel --dtype float32 --groupsize $groupsize --quant_error_title $title --block_shape_width 32 --block_shape_height 32 --outlier_threshold $thresh > out2.txt
                # python test_quantize.py facebook/opt-1.3b pajama --sparse True --wbits 3 --spbits 8 --perchannel --dtype float32 --groupsize 16 --block_shape_width 8 --block_shape_height 8 --outlier_threshold $thresh --nsamples 128 --subsample $subsamples --quant_error_title $title --block_mask_density $thresh_block > out2.txt
                grep "$VAR1 = " out2.txt >> res2.txt
                grep "$VAR2 = " out2.txt >> res2.txt
                grep "$VAR3 = " out2.txt >> res2.txt
                grep "$VAR4 = " out2.txt >> res2.txt
                grep "$VAR5 = " out2.txt >> res2.txt
                echo -e "\n" >> res2.txt
        done
done
