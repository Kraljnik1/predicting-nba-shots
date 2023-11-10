# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## UZOP Projekt - Predicting NBA shots
# #### Ana Ujević 0036534085
# # Priprema i vizualizacija podataka
#

# %% [markdown]
# #### Korišteni paketi

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ## Analiza i čišćenje podataka
# ### Učitavanje i pregled podataka

# %%
X = pd.read_csv("shot_logs.csv")
#X.shape - prikaz koliko podataka ima (128069,21)
X.head() #prikaz podataka u obliku tablice



# %% [markdown]
# ### Značajke

# %%
X.columns.values

# %% [markdown]
# ### Opis značajki
# - 'GAME_ID': Jedinstveni identifikator igre.
# - 'MATCHUP': Datum i informacije o protivnicima i tipu igre
# - LOCATION': Mjesto na kojem se igrala utakmica (npr. "H" za domaću utakmicu i "A" za gostujuću utakmicu).
# - 'W': Rezultat igre za ekipu igrača ("W" označava pobjedu, "L" označava poraz).
# - 'FINAL_MARGIN': Konačna razlika u bodovima na kraju igre.
# - 'SHOT_NUMBER': Redni broj šuta igrača tijekom utakmice.
# - 'PERIOD': Period utakmice u kojem je šut izveden (npr. 1. četvrtina, 2. četvrtina itd.).
# - 'GAME_CLOCK': Preostalo vrijeme u igri u trenutku izvođenja šuta.
# - 'SHOT_CLOCK': Preostalo vrijeme u trenutku izvođenja šuta u tom napadu (napad traje 25s)
# - 'DRIBBLES': Broj driblinga koje je igrač napravio prije šuta.
# - 'TOUCH_TIME': Ukupno vrijeme dodira s loptom prije izvođenja šuta.
# - 'SHOT_DIST': Udaljenost igrača od koša u trenutku izvođenja šuta.
# - 'PTS_TYPE': Vrijednost šuta (npr. "2" za dvostruki šut i "3" za trica).
# - 'SHOT_RESULT': Ishod šuta (npr. "made" za uspješan šut i "missed" za promašen šut).
# - 'CLOSEST_DEFENDER': Ime najbližeg braniča protivničke momčadi.
# - 'CLOSEST_DEFENDER_PLAYER_ID': Jedinstveni identifikator najbližeg braniča.
# - 'CLOSE_DEF_DIST': Udaljenost između igrača i najbližeg braniča u trenutku izvođenja šuta.
# - 'FGM': Oznaka za uspješan šut ("1" za uspješan šut, "0" za promašaj).
# - 'PTS': Broj bodova osvojenih šutom.
# - 'player_name': Ime igrača koji je izveo šut.
# - 'player_id': Jedinstveni identifikator igrača koji baca
#   dentifikator igrača.

# %%
X.info() # informacije o tipovima podataka značajki

# %% [markdown]
# ### Čišćenje podataka

# %%
X.describe()

# %% [markdown]
# ### Monotoni atributi 

# %% [markdown]
# - provjeravamo postoje li monotoni atributi, odnosno atributi čija vrijednost jednoliko raste

# %%
X.nunique() #broji jedinstvene vrijednosti po stupcima


# %% [markdown]
# - zaključujemo da su monotoni atributi: game_id, player_id i matchup te ih izbacujemo

# %%
X.drop(('GAME_ID'), axis=1, inplace=True)
X.drop(('player_id'), axis=1, inplace=True)
X.drop(('MATCHUP'), axis=1, inplace=True)
X.drop(('player_name'), axis=1, inplace=True)

# %%
#prikaz nakon izbacivanja
X.nunique() 

# %% [markdown]
# ### Nedostajući podaci

# %%
X.isna().sum() #nedostajuci podaci, u primjeru nemamo vrijednost za to

# %% [markdown]
# - vidimo da nedostaju podaci za shot_clock -> izbacujemo te zapise iz skupa podataka (5000 vs 128 000, broj primjera nije od velike vaznosti)

# %%
X_tmp = X.copy()
X_tmp = X_tmp.loc[X_tmp.SHOT_CLOCK.notnull(), :] ##izbacili smo primjere koji imaju vrijednost vremena koje je preostalo
X = X_tmp

# %%
X.isna().sum()

# %% [markdown]
# ## Deskriptivna statistika podataka

# %% [markdown]
# - Deskriptivna statistika pruža osnovne informacije o raspodjeli, srednjim vrijednostima, varijabilnosti i drugim karakteristikama podataka
# - nužna je za dobro razumijevanja dataseta te identificiranje stršećih vrijednosti, analizu raspodjele i pripremu za daljnje analize,
# uključujući izradu grafova

# %% [markdown]
# ## Stršeći podaci

# %%
X.describe()

# %% [markdown]
# - primjećujemo da je min(Final_margin - razlika u rezultatu) ima negativnu vrijednost -> možemo znati je li domaći tim dobio ili izgubio, ali upitna je smisao zbog te nagativne vrijednosti
# - touch_time ima negativnu vrijednost
# - primjećujemo da značajka period ima čudnu max vrijednost 7, a moguće je samo 4

# %% [markdown]
# # Vizualizacija podataka
# ## Vizualizacija stršećih vrijednosti

# %% [markdown]
# - vizualiziramo stršeću vrijednost za značajku TOUCH_TIME

# %%
#'TOUCH_TIME': Ukupno vrijeme dodira s loptom prije izvođenja šuta.
sns.boxplot(X['TOUCH_TIME']) 
plt.show()


# %% [markdown]
# - izbacujemo sve koji imaju negativnu vrijednost ukupnog dodira s loptom prije šuta

# %%
X = X.loc[(X.TOUCH_TIME > 0)]
sns.boxplot(X['TOUCH_TIME'])

# %% [markdown]
# - vizualiziramo podatke za PERIOD, odnosno period utakmice u kojem je šut izveden (npr. 1. četvrtina, 2. četvrtina itd.)

# %%
sns.boxplot(X['PERIOD']) 
plt.show()

# %% [markdown]
# - znamo da postoji ukupno 4 perioda te izbacujemo sve primjere gdje je period > 4

# %%
X = X.loc[(X.PERIOD <= 4)]
sns.boxplot(X['PERIOD'])

# %% [markdown]
# ## Vizualizacija korelacija

# %% [markdown]
# - korisno je vizualizirati korelaciju između svake značajke i ciljne varijable, kao što je 'SHOT_RESULT' ili FGM (binarna varijabla o uspješnosti bacanja)

# %%
X_numeric = X.loc[:, ~X.columns.isin(['LOCATION', 'W', 'GAME_CLOCK', 'SHOT_RESULT', 'CLOSEST_DEFENDER'])]

# corr() funkcija za izračun korelacije između numeričkih značajki
correlation_matrix = X_numeric.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()

# %% [markdown]
# ## Histogrami
# - možemo prikazati histograme za različite značajke

# %% [markdown]
# ### SHOT_DISC histogram
# -  prikazuje  učestalost šuteva na različitim udaljenostima od koša

# %%
plt.figure(figsize=(10, 6))
sns.histplot(X['SHOT_DIST'], bins=20, kde=True)
plt.xlabel('Udaljenost od koša (feet)')
plt.ylabel('Broj šuteva')
plt.title('Histogram udaljenosti od koša')
plt.show()

# %% [markdown]
# ### SHOT_TIME histrogram
# -  histogram za 'TOUCH_TIME' koji prikazuje raspodjelu vremena koje igrači provode s loptom prije izvođenja šuta

# %%
plt.figure(figsize=(10, 6))
sns.histplot(X['TOUCH_TIME'], bins=20, kde=True)
plt.xlabel('Vrijeme dodira s loptom (sekunde)')
plt.ylabel('Broj šuteva')
plt.title('Histogram vremena dodira s loptom prije šuta')
plt.show()

# %% [markdown]
# ## Stupičasti dijagram
# - želimo prikazat ovisnost broja pogodaka o četvrti

# %%
period_fgm = X.groupby('PERIOD')['FGM'].sum().reset_index() #Grupiranje podatke prema četvrtima i izračun ukupan broj pogodaka (FGM) u svakoj četvrti

plt.figure(figsize=(10, 6))
sns.barplot(data=period_fgm, x='PERIOD', y='FGM')
plt.xlabel('Četvrt')
plt.ylabel('Ukupan broj pogodaka (FGM)')
plt.title('Ovisnost broja pogodaka o četvrti')
plt.show()


# %% [markdown]
# ## Linijski dijagram
# - želimo prikazati raspodjelu pogodaka u odnosu na udaljenost od koša
# - ovo je gore napravljeni dijagram samo na linijski način

# %%
X['SHOT_DIST_METERS'] = X['SHOT_DIST'] * 0.3048 #pretvraamo stopala u metre
distance_fg_data = X.groupby('SHOT_DIST_METERS')['FGM'].sum().reset_index() # grupiranje podatka po udaljenosti i zatim sumirati broj pogodaka za svaku grupu

plt.figure(figsize=(10, 6))
sns.lineplot(data=distance_fg_data, x='SHOT_DIST_METERS', y='FGM', marker='o')
plt.xlabel('Udaljenost od koša (m)')
plt.ylabel('Ukupan broj pogodaka (FGM)')
plt.title('Raspodjela pogodaka po udaljenosti od koša')
plt.show()

# %% [markdown]
# ## Matrica dijagrama raspršenja

# %%
sns.set(rc={'figure.figsize':(25,15)})
sns.pairplot(X.loc[:,['FINAL_MARGIN', 'SHOT_NUMBER', 'SHOT_CLOCK', 'DRIBBLES', 'TOUCH_TIME', 'SHOT_DIST', 'SHOT_RESULT']], hue="SHOT_RESULT", corner=True)
plt.show()

# %% [markdown]
# ## Zadnji pregled pogodataka
# - U znanstvenom radu piše da su FGM i PTS savršeni prediktori shot resulta što znači da možemo izbaciti bilo koje dvije od te
# tri značajke za daljni rad s modelom

# %%
X.drop(('FGM'), axis=1, inplace=True)
X.drop(('PTS'), axis=1, inplace=True)
X.drop(('PTS_TYPE'), axis=1, inplace=True) #izbacujem jer direktno ovisi o SHOT_DIST

# %%
X. describe()

# %%
