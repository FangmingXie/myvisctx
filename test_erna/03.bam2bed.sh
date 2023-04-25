#!/bin/bash

$input="./test_erna_data/P8_test.bam"
$output="./test_erna_data/P8_test.v1.bed"

samtools view $input | cut -f 3,4 | awk 'BEGIN{FS=OFS="\t"}{print $1, $2-1, $2-1+89}' > $output