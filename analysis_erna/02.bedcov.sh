

datadir="./test_erna_data/"
# non gene vs gene
samtools bedcov -Q 30 "$datadir"allgenebodies_complement.bed "$datadir"P8_test.bam > "$datadir"bedcov_v2.nongene
samtools bedcov -Q 30 "$datadir"allgenebodies.bed "$datadir"P8_test.bam > "$datadir"bedcov_v2.gene