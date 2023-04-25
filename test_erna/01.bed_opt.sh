#!/bin/bash 

input="./test_erna_data/allgenebodies.bed"
genome="./test_erna_data/mm10.sorted.chrom.sizes"
output="./test_erna_data/allgenebodies_complement.bed"

bedtools complement -i $input -g $genome > $output