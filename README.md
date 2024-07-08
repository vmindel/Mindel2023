# Mindel2023

`conda create -name env_name python=3.8.3`

`conda activate env_name`

`pip install -r requirements.txt`

`python -m ipykernel install --user --name env_name --display-name 2023vosa`

## In terminal creater all needed folders to save the figures

`mkdir -p figures/Main/{fig1,fig2,fig3,fig4,fig5,fig6,fig7} figures/Supp/`
## I provided a Snakemake file which aggregates all of the notebooks executes them and saves the figures in the needed locations to run the worfklow:
`snakemake all --cores 16`
