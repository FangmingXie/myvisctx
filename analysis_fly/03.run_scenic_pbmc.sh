#!/bin/bash

nprocs=4

# transcription factors list
f_tfs="/scr/qlyu/projects/v1/data/annot/allTFs_hg38.txt"
# ranking databases
f_db_names='/scr/qlyu/projects/v1/data/annot/hg38__refseq-r80__10kb_up_and_down_tss.mc9nr.genes_vs_motifs.rankings.feather'
# motif databases
f_motif_path="/scr/qlyu/projects/v1/data/annot/motifs-v9-nr.hgnc-m0.001-o0.0.tbl"

outdir="/scr/qlyu/projects/v1/data/10x_pbmc/res"
f_loom_path_scenic="$outdir/pbmc10k_filtered_scenic.loom"
f_pyscenic_output="$outdir/pyscenic_output_v2.loom"
f_final_loom="$outdir/pbmc10k_scenic_integrated-output_v2.loom"
f_adj="$outdir/adj_v2.csv"
f_reg="$outdir/reg_v2.csv"
f_umap="$outdir/scenic_umap_v2.txt" 
f_tsne="$outdir/scenic_tsne_v2.txt"

# # step 1. (4 hrs)
pyscenic grn ${f_loom_path_scenic} ${f_tfs} -o ${f_adj} --num_workers $nprocs

# step 2-3. (0.5 hr)
pyscenic ctx ${f_adj} \
    ${f_db_names} \
    --annotations_fname ${f_motif_path} \
    --expression_mtx_fname ${f_loom_path_scenic} \
    --output ${f_reg} \
    --mask_dropouts \
    --num_workers $nprocs

# step 4. (10 min)
pyscenic aucell \
    ${f_loom_path_scenic} \
    ${f_reg} \
    --output ${f_pyscenic_output} \
    --num_workers $nprocs
