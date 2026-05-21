#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 path/to/report.txt"
}

if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi

input=$1
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ ! -f "$input" && -f "$script_dir/$input" ]]; then
    input="$script_dir/$input"
fi

if [[ ! -f "$input" ]]; then
    echo "error: input file not found: $1" >&2
    exit 1
fi

case "$input" in
    *.txt) ;;
    *)
        echo "error: input file must end in .txt: $input" >&2
        exit 1
        ;;
esac

input_dir=$(cd -- "$(dirname -- "$input")" && pwd)
input_file=$(basename -- "$input")
base=${input_file%.txt}
tex_file="$input_dir/$base.tex"

cat > "$tex_file" <<EOF
\documentclass{article}
\usepackage{amsmath,amssymb}
\usepackage[T1]{fontenc}

\begin{document}
\input{$input_file}
\end{document}
EOF

(
    cd "$input_dir"
    if command -v pdflatex >/dev/null 2>&1; then
        latexmk -pdf "$base.tex"
    else
        tectonic_bin=${TECTONIC:-}
        if [[ -z "$tectonic_bin" ]]; then
            if command -v tectonic >/dev/null 2>&1; then
                tectonic_bin=tectonic
            elif [[ -x "$HOME/.conda/envs/latex-tectonic/bin/tectonic" ]]; then
                tectonic_bin="$HOME/.conda/envs/latex-tectonic/bin/tectonic"
            elif [[ -x /tmp/latex-tectonic/bin/tectonic ]]; then
                tectonic_bin=/tmp/latex-tectonic/bin/tectonic
            fi
        fi

        if [[ -z "$tectonic_bin" ]]; then
            echo "error: pdflatex not found, and no tectonic fallback is available" >&2
            echo "hint: install texlive-latex-base or install tectonic and put it on PATH" >&2
            exit 1
        fi

        XDG_CACHE_HOME=${XDG_CACHE_HOME:-$HOME/.cache} \
            "$tectonic_bin" --keep-logs --keep-intermediates "$base.tex"
    fi
)

echo "wrote $tex_file"
echo "wrote $input_dir/$base.pdf"
