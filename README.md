# UZOP Projekt (Predicting NBA Shots) - Deskriptivna statistika

- Nikola Kraljević
___

## Upute:

Koristit cemo 3.8 verziju pythona jer se to trazi na strojnom ucenju, te pip umjesto conde.
```
>python --version
Python 3.8.18

>pip --version
pip 23.2.1 
```

Koristit cemo paket _jupytext_ da imamo citljive commitove(u markdownu) i lakse verzioniranje.


#### Koristeni paketi:

```
pip install scipy
pip install jupyter
pip install pandas
pip install jupytext
```

___

## Jupytext:

### How to pair notebook:

U JupyterLab([http://localhost:8888/doc/tree/SU1-2023-LAB1.ipynb]) otvoriti _View_ > _Activate Command Pallete_ ili sa _Ctrl + Shift + c_ 

U search baru treba ukucat "_pair_" i odabrat *Pair Notebook with Markdown*

Kada se to napravilo imamo dvije verzije notebooka u .ipynb i u .md formatu, .md format je *human readable*!

___

## Inicijalizacija venv (optional)

Nakon cloneanja potrebno je pokrenuti ovu naredbu:
```
cd predicting-nba-shots
python3.8 -m venv env
```

Nakon sto se kreirao _env/_ folder: 
```
source env/bin/activate
```

Instalacija paketa iz _requirements.txt_:
```
pip install -r requirements.txt
```
