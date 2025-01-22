#!/bin/bash
VAR1="wikitext2 perplexity"
VAR2="ptb perplexity"
VAR3="c4 perplexity"
VAR4="total_gb_size"
groupsize=16
thresh=0.09
for spbits in 3 4 8 16;
do
    for wbits in 3 4 8;
    do
        echo -e "$wbits/$spbits, $groupsize, $thresh" >> res.txt
        python test_quantize.py facebook/opt-1.3b /home/lj9979/SpQR/data/red_pajama_n=1024_2048_context_length.pth --sparse True --wbits $wbits --spbits $spbits --density 0.1 --gpu 0 --perchannel --dtype float32 --groupsize $groupsize --block_shape_width 32 --block_shape_height 32 --outlier_threshold $thresh > out.txt
        grep "$VAR1 = " out.txt >> res.txt
        grep "$VAR2 = " out.txt >> res.txt
        grep "$VAR3 = " out.txt >> res.txt
        grep "$VAR4 = " out.txt >> res.txt
        echo -e "\n" >> res.txt
    done
done
