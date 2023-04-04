#!/bin/bash

nprocs=4

# transcription factors list
f_tfs="/scr/qlyu/projects/v1/data/annot/allTFs_dmel.txt"
# ranking databases
f_db_names='/scr/qlyu/projects/v1/data/annot/dm6_v10_clust.genes_vs_motifs.rankings.feather'
# motif databases
f_motif_path="/scr/qlyu/projects/v1/data/annot/motifs-v10nr_clust-nr.flybase-m0.001-o0.0.tbl"

outdir="/scr/qlyu/projects/v1/data/jain22_nature_flybrain/res"
f_loom="$outdir/scEcRDN.loom"
f_pyscenic_output="$outdir/pyscenic_output.loom"
f_adj="$outdir/adj.csv"
f_reg="$outdir/reg.csv"
f_umap="$outdir/scenic_umap.txt" 
f_tsne="$outdir/scenic_tsne.txt"

# # step 1. (4 hrs)
pyscenic grn ${f_loom} ${f_tfs} -o ${f_adj} --num_workers $nprocs

# step 2-3. (0.5 hr)
pyscenic ctx ${f_adj} \
    ${f_db_names} \
    --annotations_fname ${f_motif_path} \
    --expression_mtx_fname ${f_loom} \
    --output ${f_reg} \
    --mask_dropouts \
    --num_workers $nprocs

# step 4. (10 min)
pyscenic aucell \
    ${f_loom} \
    ${f_reg} \
    --output ${f_pyscenic_output} \
    --num_workers $nprocs
