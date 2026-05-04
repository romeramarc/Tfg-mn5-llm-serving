# Compila memoria-tfg.tex desde esta carpeta (usa latexmk si existe).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (Get-Command latexmk -ErrorAction SilentlyContinue) {
    latexmk -pdf -interaction=nonstopmode memoria-tfg.tex
} else {
    pdflatex -interaction=nonstopmode memoria-tfg.tex
    pdflatex -interaction=nonstopmode memoria-tfg.tex
}
