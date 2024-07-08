import os
import pandas as pd
import glob

IDS = glob.glob('*.ipynb')
IDS = [name.split('.')[0] for name in IDS]

rule all:
    input:
        expand('res_notebooks/{notebook}.ipynb', notebook=IDS) 

rule run_notebook:
    input: 
        '{notebook}.ipynb',
    output:
        'res_notebooks/{notebook}.ipynb'
    shell:
        'papermill {input} {output} -k 2023vosa'
