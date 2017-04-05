#!/bin/bash

platex fog.tex
dvipdfmx fog.dvi 

platex fog_min.tex
dvipdfmx fog_min.dvi 

platex h3_spec_agg.tex
dvipdfmx h3_spec_agg.dvi 

platex kripke.tex
dvipdfmx kripke.dvi 

platex lightweight.tex
dvipdfmx lightweight.dvi 

platex whiteline.tex
dvipdfmx whiteline.dvi 

platex x5.tex
dvipdfmx x5.dvi 

