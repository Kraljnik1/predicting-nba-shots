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
#sns.set(rc={'figure.figsize':(25,15)})
#sns.pairplot(X.loc[:,['FINAL_MARGIN', 'SHOT_NUMBER', 'SHOT_CLOCK', 'DRIBBLES', 'TOUCH_TIME', 'SHOT_DIST', 'SHOT_RESULT']], hue="SHOT_RESULT", corner=True)
#plt.show()

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

# %% [markdown]
# # Replikacija rezultata

# %% [markdown]
# ### - dodatno čićenje

# %%
#Convert game clock to seconds
#X['GAME_CLOCK'] = X['GAME_CLOCK'].apply(
  #  lambda x: 60*int(x.split(':')[0]) + int(x.split(':')[1]))

# %%
X.describe()

# %% [markdown]
# ## 1. Logistička regresija

# %% [markdown]
# - Logistička regresija generira linearni set težina θ
# - Mičem kategoričke varijable koje imaju više od dvije kategorije - to uključuje "GAME ID", "MATCHUP", "CLOSEST DEFENDER", "CLOSEST DEFENDER PLAYER ID", 
# "player name" i "player id". Razlog je što kada sam zamijenila kategorije s cjelobrojnim vrijednostima te 
# bi mogle zbuniti model koji pokušava naučiti linearne težine.

# %%
print(X.columns)

# %%
# Pretvaranje varijable X u DataFrame T
T = pd.DataFrame(X)

# %%
print(T.columns)

# %%
# Uklanjanje određenih stupaca iz tablice
columns_to_drop = ['CLOSEST_DEFENDER_PLAYER_ID']
T = T.drop(columns=columns_to_drop)
T.describe()

# %%
T.head()


# %%
# Pretvaranje vremena u sekunde
def calc_secs(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return (12 * 60 - 60 * hours + minutes)

T['PERIOD'] = (T['PERIOD'] - 1) * (12 * 60)
T['GAME_CLOCK'] = T['GAME_CLOCK'].apply(calc_secs)
T['GAME_TIME'] = T['PERIOD'] + T['GAME_CLOCK']
T = T.drop(columns=['PERIOD', 'GAME_CLOCK', 'CLOSEST_DEFENDER' ])

# Pretvaranje kategoričkih varijabli u numeričke
T['LOCATION'] = T['LOCATION'].apply(lambda x: 1 if x == 'A' else -1)
T['W'] = T['W'].apply(lambda x: 1 if x == 'W' else -1)
T['SHOT_RESULT'] = T['SHOT_RESULT'].apply(lambda x: 1 if x == 'made' else -1)

# %%
T.head()

# %% [markdown]
# - kombinirala sam značajku PERIOD i GAME_CLOCK  u jednu značajku: "UKUPNO VRIJEME IGRE". 
# - "GAME CLOCK" značajka odbrojava i resetira se nakon svakog perioda, ali ona mora biti takva da razlikuje 10 sekundi preostalih u prvom periodu
# od 10 sekundi preostalih u četvrtom periodu ( kada su igrači umorniji i možda skloniji promašiti šut).
#

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Provjera postojanja kolone 'SHOT_RESULT'
if 'SHOT_RESULT' in T.columns:
    # Izdvajanje stupca SHOT_RESULT u varijablu y
    y = T['SHOT_RESULT']

    # Makni stupac SHOT_RESULT iz dataseta T
    T.drop(columns=['SHOT_RESULT'], inplace=True)

    # Provjera nedostajućih vrednosti u podacima
    if T.isnull().values.any() or y.isnull().values.any():
        print("Postoje nedostajuće vrednosti u podacima")
    else:
        # Podijeli podatke T i y u omjeru 50:50 za trening i test skup
        T_train, T_test, Y_train, Y_test = train_test_split(T, y, test_size=0.5, random_state=42)

        # Inicijalizacija i treniranje modela s regularizacijom lambda=0.001
        clf = LogisticRegression(C=1/0.001)
        clf.fit(T_train, Y_train)

        # Predviđanje na testnom skupu
        predictions = clf.predict(T_test)

        # Računanje confusion matrix
        cm = confusion_matrix(Y_test, predictions)
        print("Confusion Matrix:")
        print(cm)

        # Računanje accuracy score
        accuracy = accuracy_score(Y_test, predictions)
        print("Accuracy Score:", accuracy)
else:
    print("Kolona 'SHOT_RESULT' nije pronađena")

# %% [markdown]
# ## 2. Naivni Bayesov klasifikator

# %% [markdown]
# - Naivni Bayes je relativno jednostavan algoritam koji pretpostavlja da su osobine xi uvjetno nezavisne uz uvjet y
# - Podatke kontinuiranih vrijedosti podijeljene su u 4 kategorije:
#     - vise od 1 standardna devijacija ispod prosjeka
#     - 1 standardna devijacija ispod prosjeka
#     - 1 standardna devijacija iznad prosjeka,
#     - vise od 1 standardna devijacija iznad prosjeka. 
# - Nažalost, ta podjela nije pokazala primjetnu razliku u rezultatima koristeći Naivni Bayes.
# - Binning -> Ova funkcija radi s kontinuiranim podacima poput brojeva, primjerice, duljina šuteva u
#   košarci ili vrijeme provedeno s loptom. Ona uzima te brojeve i sortira ih u četiri kategorije kako bi Naivni Bayes model mogao bolje razumjeti i naučiti iz tih grupa umjesto iz točnih vrijednosti.
#
#
#

# %%
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Podaci su već definirani u varijablama T i y
# T i y su vaši stvarni podaci koje želite podijeliti na trening i test skupove

# Podijela podataka na trening i test skupove
T_train, T_test, Y_train, Y_test = train_test_split(T, y, test_size=0.5)

# Funkcija za biniranje kontinuiranih varijabli u 4 kategorije

def bin_continuous_data(data):
    data_abs = np.abs(data)
    means = np.mean(data_abs, axis=0)
    stds = np.std(data_abs, axis=0)

    boundaries = np.array([means - stds, means - 0.5 * stds, means + 0.5 * stds, means + stds])

    categories = np.zeros_like(data, dtype=int)
    for i in range(data.shape[1]):
        for j in range(4):
            categories[:, i] += (data[:, i] > boundaries[j, i])

    return categories

# Binarizacija podataka
T_train_binned = bin_continuous_data(T_train.values)
T_test_binned = bin_continuous_data(T_test.values)

# Treniranje multinomijalnog Naivnog Bayesa
clf = MultinomialNB(alpha=1.0)
clf.fit(T_train_binned, Y_train)

# Cross-validation
scores = cross_val_score(clf, T_train_binned, Y_train, cv=5)
print("Cross-validation scores:", scores)

# Predviđanje na testnom skupu
predictions = clf.predict(T_test_binned)

# Računanje točnosti
accuracy = accuracy_score(Y_test, predictions)
print('Test accuracy:', accuracy)


conf_matrix = confusion_matrix(Y_test, predictions)
print('Confusion Matrix:')
print(conf_matrix)

# %% [markdown]
# ## 3. Random Forest klasifikator

# %% [markdown]
# - Random Forest klasiifikator dobro obrađuju kategoričke podatke zbog prirode stabala odlučivanja.
# Također je otporan  na prenaučenost jer koristi više stabala odlučivanja za klasifikaciju podataka.
# - algoritam ansambla koji nasumično bira podskup značajki, stvara optimalno stablo odlučivanja od tih značajki, zatim ponavlja proces s novim podskupovima značajki kako bi na kraju stvorio "šumu" stabala odlučivanja

# %%
Z = pd.read_csv("shot_logs.csv")
Z.drop(('GAME_ID'), axis=1, inplace=True)
Z.drop(('player_id'), axis=1, inplace=True)
Z.drop(('MATCHUP'), axis=1, inplace=True)
Z.drop(('player_name'), axis=1, inplace=True)

X_tmp = Z.copy()
X_tmp = X_tmp.loc[X_tmp.SHOT_CLOCK.notnull(), :] ##izbacili smo primjere koji imaju vrijednost vremena koje je preostalo
Z = X_tmp

Z.drop(('FGM'), axis=1, inplace=True)
Z.drop(('PTS'), axis=1, inplace=True)
Z.drop(('PTS_TYPE'), axis=1, inplace=True) #izbacujem jer direktno ovisi o SHOT_DIST

T = pd.DataFrame(Z)
T.columns
# Uklanjanje određenih stupaca iz tablice
columns_to_drop = ['CLOSEST_DEFENDER_PLAYER_ID']
T = T.drop(columns=columns_to_drop)
T.describe()
# Pretvaranje vremena u sekunde
def calc_secs(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return (12 * 60 - 60 * hours + minutes)

T['PERIOD'] = (T['PERIOD'] - 1) * (12 * 60)
T['GAME_CLOCK'] = T['GAME_CLOCK'].apply(calc_secs)
T['GAME_TIME'] = T['PERIOD'] + T['GAME_CLOCK']
T = T.drop(columns=['PERIOD', 'GAME_CLOCK', 'CLOSEST_DEFENDER' ])

# Pretvaranje kategoričkih varijabli u numeričke
T['LOCATION'] = T['LOCATION'].apply(lambda x: 1 if x == 'A' else -1)
T['W'] = T['W'].apply(lambda x: 1 if x == 'W' else -1)
T['SHOT_RESULT'] = T['SHOT_RESULT'].map({'made': 1, 'missed': -1})
T

# %%
T

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.metrics import confusion_matrix, accuracy_score

# Provjera postojanja kolone 'SHOT_RESULT'
if 'SHOT_RESULT' in T.columns:
    # Izdvajanje stupca SHOT_RESULT u varijablu y
    y = T['SHOT_RESULT']

    # Makni stupac SHOT_RESULT iz dataseta T
    T.drop(columns=['SHOT_RESULT'], inplace=True)

    # Provjera nedostajućih vrednosti u podacima
    if T.isnull().values.any() or y.isnull().values.any():
        print("Postoje nedostajuće vrednosti u podacima")
    else:
        # Podijeli podatke T i y u omjeru 50:50 za trening i test skup
        T_train, T_test, Y_train, Y_test = train_test_split(T, y, test_size=0.5, random_state=42)

        clf = RandomForestClassifier()
        clf.fit(T_train, Y_train)

        # Predviđanje na testnom skupu
        predictions = clf.predict(T_test)

        # Računanje confusion matrix
        cm = confusion_matrix(Y_test, predictions)
        print("Confusion Matrix:")
        print(cm)

        # Računanje accuracy score
        accuracy = accuracy_score(Y_test, predictions)
        print("Accuracy Score:", accuracy)
else:
    print("Kolona 'SHOT_RESULT' nije pronađena")

# %% [markdown]
# ## 4. SVM

# %%
Z = pd.read_csv("shot_logs.csv")
Z.drop(('GAME_ID'), axis=1, inplace=True)
Z.drop(('player_id'), axis=1, inplace=True)
Z.drop(('MATCHUP'), axis=1, inplace=True)
Z.drop(('player_name'), axis=1, inplace=True)

X_tmp = Z.copy()
X_tmp = X_tmp.loc[X_tmp.SHOT_CLOCK.notnull(), :] ##izbacili smo primjere koji imaju vrijednost vremena koje je preostalo
Z = X_tmp

Z.drop(('FGM'), axis=1, inplace=True)
Z.drop(('PTS'), axis=1, inplace=True)
Z.drop(('PTS_TYPE'), axis=1, inplace=True) #izbacujem jer direktno ovisi o SHOT_DIST

T = pd.DataFrame(Z)
T.columns
# Uklanjanje određenih stupaca iz tablice
columns_to_drop = ['CLOSEST_DEFENDER_PLAYER_ID']
T = T.drop(columns=columns_to_drop)
T.describe()
# Pretvaranje vremena u sekunde
def calc_secs(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return (12 * 60 - 60 * hours + minutes)

T['PERIOD'] = (T['PERIOD'] - 1) * (12 * 60)
T['GAME_CLOCK'] = T['GAME_CLOCK'].apply(calc_secs)
T['GAME_TIME'] = T['PERIOD'] + T['GAME_CLOCK']
T = T.drop(columns=['PERIOD', 'GAME_CLOCK', 'CLOSEST_DEFENDER' ])

# Pretvaranje kategoričkih varijabli u numeričke
T['LOCATION'] = T['LOCATION'].apply(lambda x: 1 if x == 'A' else -1)
T['W'] = T['W'].apply(lambda x: 1 if x == 'W' else -1)
T['SHOT_RESULT'] = T['SHOT_RESULT'].map({'made': 1, 'missed': -1})
T

# %%
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

# Provjera postojanja kolone 'SHOT_RESULT'
if 'SHOT_RESULT' in T.columns:
    # Izdvajanje stupca SHOT_RESULT u varijablu y
    y = T['SHOT_RESULT']

    # Makni stupac SHOT_RESULT iz dataseta T
    T.drop(columns=['SHOT_RESULT'], inplace=True)

    # Provjera nedostajućih vrednosti u podacima
    if T.isnull().values.any() or y.isnull().values.any():
        print("Postoje nedostajuće vrednosti u podacima")
    else:
        # Podijeli podatke T i y u omjeru 50:50 za trening i test skup
        T_train, T_test, Y_train, Y_test = train_test_split(T, y, test_size=0.5, random_state=42)

        clf = make_pipeline(StandardScaler(),LinearSVC(dual=False, random_state=0, tol=1e-5))
        clf.fit(T_train, Y_train)

        # Predviđanje na testnom skupu
        predictions = clf.predict(T_test)

        # Računanje confusion matrix
        cm = confusion_matrix(Y_test, predictions)
        print("Confusion Matrix:")
        print(cm)

        # Računanje accuracy score
        accuracy = accuracy_score(Y_test, predictions)
        print("Accuracy Score:", accuracy)
else:
    print("Kolona 'SHOT_RESULT' nije pronađena")

# %% [markdown]
# ## 5. Neuralne mreže

# %%
Z = pd.read_csv("shot_logs.csv")
Z.drop(('GAME_ID'), axis=1, inplace=True)
Z.drop(('player_id'), axis=1, inplace=True)
Z.drop(('MATCHUP'), axis=1, inplace=True)
Z.drop(('player_name'), axis=1, inplace=True)

X_tmp = Z.copy()
X_tmp = X_tmp.loc[X_tmp.SHOT_CLOCK.notnull(), :] ##izbacili smo primjere koji imaju vrijednost vremena koje je preostalo
Z = X_tmp

Z.drop(('FGM'), axis=1, inplace=True)
Z.drop(('PTS'), axis=1, inplace=True)
Z.drop(('PTS_TYPE'), axis=1, inplace=True) #izbacujem jer direktno ovisi o SHOT_DIST

T = pd.DataFrame(Z)
T.columns
# Uklanjanje određenih stupaca iz tablice
columns_to_drop = ['CLOSEST_DEFENDER_PLAYER_ID']
T = T.drop(columns=columns_to_drop)
T.describe()
# Pretvaranje vremena u sekunde
def calc_secs(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return (12 * 60 - 60 * hours + minutes)

T['PERIOD'] = (T['PERIOD'] - 1) * (12 * 60)
T['GAME_CLOCK'] = T['GAME_CLOCK'].apply(calc_secs)
T['GAME_TIME'] = T['PERIOD'] + T['GAME_CLOCK']
T = T.drop(columns=['PERIOD', 'GAME_CLOCK', 'CLOSEST_DEFENDER' ])

# Pretvaranje kategoričkih varijabli u numeričke
T['LOCATION'] = T['LOCATION'].apply(lambda x: 1 if x == 'A' else -1)
T['W'] = T['W'].apply(lambda x: 1 if x == 'W' else -1)
T['SHOT_RESULT'] = T['SHOT_RESULT'].map({'made': 1, 'missed': -1})
T

# %%
y = T['SHOT_RESULT']

# Makni stupac SHOT_RESULT iz dataseta T
T.drop(columns=['SHOT_RESULT'], inplace=True)

# %%
#Treniranje modela

# standardizacija
scaler = StandardScaler()
scaler.fit(T)
T = scaler.transform(T)
T

# %%
T

# %%
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


relu_scores = np.array([])
sigmoid_scores = np.array([])


T_train, T_test, Y_train, Y_test = train_test_split(T, y, stratify=y)

# default alpha je 0.0001
clf_relu = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes=[50], alpha=0.001).fit(T_train, Y_train) # RELU aktivacijska funkcija
clf_sigmoid = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes=[50], activation="logistic", alpha=0.001).fit(T_train, Y_train) # RELU aktivacijska funkcija!

relu_scores = np.append(relu_scores, clf_relu.score(T_test, Y_test))
sigmoid_scores = np.append(sigmoid_scores, clf_sigmoid.score(T_test, Y_test))

print(f"relu_scores average: {np.average(relu_scores)}")
print(f"sigmoid_scores average: {np.average(sigmoid_scores)}")

T_train, T_test, Y_train, Y_test = train_test_split(T, y, stratify=y, test_size=0.1) 
clf = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes=[50], activation="logistic").fit(T_train, Y_train) 
y_pred = clf.predict(T_test)
print(confusion_matrix(Y_test,y_pred))
print(clf.score(T_test, Y_test))

# %% [markdown]
# ## 6. XGBOOST

# %%
Z = pd.read_csv("shot_logs.csv")
Z.drop(('GAME_ID'), axis=1, inplace=True)
Z.drop(('player_id'), axis=1, inplace=True)
Z.drop(('MATCHUP'), axis=1, inplace=True)
Z.drop(('player_name'), axis=1, inplace=True)

X_tmp = Z.copy()
X_tmp = X_tmp.loc[X_tmp.SHOT_CLOCK.notnull(), :] ##izbacili smo primjere koji imaju vrijednost vremena koje je preostalo
Z = X_tmp

Z.drop(('FGM'), axis=1, inplace=True)
Z.drop(('PTS'), axis=1, inplace=True)
Z.drop(('PTS_TYPE'), axis=1, inplace=True) #izbacujem jer direktno ovisi o SHOT_DIST

T = pd.DataFrame(Z)
T.columns
# Uklanjanje određenih stupaca iz tablice
columns_to_drop = ['CLOSEST_DEFENDER_PLAYER_ID']
T = T.drop(columns=columns_to_drop)
T.describe()
# Pretvaranje vremena u sekunde
def calc_secs(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return (12 * 60 - 60 * hours + minutes)

T['PERIOD'] = (T['PERIOD'] - 1) * (12 * 60)
T['GAME_CLOCK'] = T['GAME_CLOCK'].apply(calc_secs)
T['GAME_TIME'] = T['PERIOD'] + T['GAME_CLOCK']
T = T.drop(columns=['PERIOD', 'GAME_CLOCK', 'CLOSEST_DEFENDER' ])

# Pretvaranje kategoričkih varijabli u numeričke
T['LOCATION'] = T['LOCATION'].apply(lambda x: 1 if x == 'A' else -1)
T['W'] = T['W'].apply(lambda x: 1 if x == 'W' else -1)
T['SHOT_RESULT'] = T['SHOT_RESULT'].map({'made': 1, 'missed': 0})
T = T.drop(columns=['LOCATION',	'W', 'FINAL_MARGIN', 'SHOT_NUMBER', 'DRIBBLES', 'GAME_TIME' ])
T

# %%

# Pretvaranje negativnih vrijednosti u NaN
T.loc[T['TOUCH_TIME'] < 0, 'TOUCH_TIME'] = np.nan

# Dropanje redova s NaN vrijednostima u stupcu TOUCH_TIME
T.dropna(subset=['TOUCH_TIME'], inplace=True)

y = T['SHOT_RESULT']

# Makni stupac SHOT_RESULT iz dataseta T
T.drop(columns=['SHOT_RESULT'], inplace=True)

# %%
T.describe()

# %%
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error, precision_score
from sklearn.model_selection import train_test_split

#izvori koje sam koristila
# https://www.kaggle.com/code/stuarthallows/using-xgboost-with-scikit-learn
# https://www.kaggle.com/code/pablocastilla/predict-if-a-shot-is-made

T_train, T_test, Y_train, Y_test = train_test_split(T, y, stratify=y, test_size=0.1)

xgb_model = XGBClassifier(objective="binary:logistic").fit(T_train, Y_train)

y_pred = xgb_model.predict(T_test)

print(confusion_matrix(Y_test, y_pred))
print(precision_score(Y_test, y_pred))

# %% [markdown]
# ## Optimizacija hiperparametara
# - cilj nam je postici accuracy bliže 68%

# %%
from sklearn.model_selection import GridSearchCV

#pretraživanje rešetke
T_train, T_test, Y_train, Y_test = train_test_split(T, y, stratify=y, test_size=0.5)
T_validation, T_test, Y_validation, Y_test = train_test_split(T, y, test_size=0.5, random_state=42)

parameters_for_testing = {
    'min_child_weight':[0.0001,0.001,0.01],
    'eta':[0.00001,0.0001,0.001],
    'n_estimators':[1,3,5,10],
    'max_depth':[3,4]
}

xgb = XGBClassifier(objective="binary:logistic")

grid_search = GridSearchCV(estimator = xgb, param_grid = parameters_for_testing, scoring='accuracy')
grid_search.fit(T_train, Y_train)

print(f'najbolji parametri {grid_search.best_params_}')
print(f'najbolji score {grid_search.best_score_}')

# %%
