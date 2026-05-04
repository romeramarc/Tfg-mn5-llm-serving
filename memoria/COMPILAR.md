# Compilar la memoria (LaTeX)

## Requisitos

En tu máquina ya aparece **TinyTeX** (TeX Live 2025) con `pdflatex` y `latexmk`. Si en otro PC no tienes LaTeX, instala [TinyTeX](https://yihui.org/tinytex/) o [MiKTeX](https://miktex.org/) y asegúrate de que `pdflatex` está en el `PATH`.

## Comando recomendado (como en Overleaf)

Desde la carpeta `memoria/`:

```bash
latexmk -pdf memoria-tfg.tex
```

`latexmk` re-ejecuta `pdflatex` las veces necesarias hasta estabilizar referencias e índices. La configuración local está en `.latexmkrc`.

### Solo con pdflatex

```bash
pdflatex -interaction=nonstopmode memoria-tfg.tex
pdflatex -interaction=nonstopmode memoria-tfg.tex
```

(Hace falta al menos dos pasadas para el índice y referencias cruzadas.)

### PowerShell en Windows

```powershell
cd memoria
.\compile.ps1
```

## Estructura modular

- `memoria-tfg.tex` — entrada principal (`\documentclass`, `\begin{document}`, `\input{...}`).
- `includes/preamble.tex` — preámbulo (paquetes y ajustes).
- `includes/00-frontmatter.tex` — portada, resúmenes, agradecimientos, índices.
- `includes/part-*.tex` — cada parte del documento.
- `includes/bibliografia.tex` — `thebibliography`.
- `memoria-tfg.monolith.tex` — copia de respaldo del documento en un solo archivo (por si quieres comparar o revertir).

Overleaf: sube **toda la carpeta `memoria/`** (incluida `includes/`) y establece como proyecto principal `memoria-tfg.tex`.

## Imágenes (`memoria/figures/`)

Pon aquí todas las figuras (PNG, JPG, PDF, etc.). En el `.tex` basta con el nombre del fichero, por ejemplo:

```latex
\includegraphics[width=0.8\textwidth]{mi_diagrama.png}
```

LaTeX busca primero en `figures/` y después en `memoria/`.

## Logo UPC

Coloca `upc.png` en **`memoria/figures/`** (preferido) o en `memoria/`. Si no existe, la portada muestra un recuadro sustituto y el PDF compila igual.

## Paquetes que faltan (TinyTeX)

Si `pdflatex` se queja de un paquete ausente:

```bash
tlmgr install nombre-del-paquete
```

## Limpieza

```bash
latexmk -c memoria-tfg.tex
```
