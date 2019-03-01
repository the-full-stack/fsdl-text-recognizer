#!/bin/bash

echo "outputting markdown..."
mkdir -p mds
pipenv run python output_markdown.py

echo "converting to pdfs..."
mkdir -p pdfs
for i in {0..13}; do pandoc "mds/$i.md" -o "pdfs/$i.pdf"; done

pdfunite pdfs/*.pdf print.pdf
rm -r mds
rm -r pdfs
echo "print.pdf is ready to be printed!"
