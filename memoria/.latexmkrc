# latexmk config (comportamiento similar a Overleaf: re-ejecuta hasta estabilizar refs)
#
# Si latexmk dice "Nothing to do" pero sale con código 12 y "pdflatex: gave an error
# in previous invocation", el estado del .fdb_latexmk está sucio: fuerza recompilación con
#   latexmk -pdf -f memoria-tfg.tex
#
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';
$max_repeat = 5;
$clean_ext = 'acn acr alg aux bbl bcf blg brf fdb_latexmk fls glg glo gls ist lof log lot out run.xml synctex.gz toc xdv';
