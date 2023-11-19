import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from collections import Counter

# Вхідні дані
data = '2 0 1 6 0 5 3 1 15 3 14 1 9 0 4 3 1 0 11 8 3 2 ' \
       '7 0 4 2 0 7 2 1 3 2 0 4 1 3 13 5 2 0 3 1 2 2 2 4 1 '\
       '1 11 2 3 2 9 11 1 8 2 3 1 0 8 1 5 4 4 2 1 4 5 10 5 3 '\
       '4 8 0 2 1 2 0 4 3 0 10 7 1 5 0 1 14 0 5 2 4 3 2 2 2 7 1 5'

print(data)
data = np.sort(np.array(data.split(' '), dtype='int8'))
print(data)
n = Counter(data)
uni_keys = list(n.keys())

# Створення DataFrame для аналізу статистичних характеристик
LABELS = ["Частоти", "Кумулятивнi частоти", 'Вiдноснi частоти', 'Кумулятивнi вiдноснi частоти']
df = pd.DataFrame(index=uni_keys, columns=LABELS)
df.index.name = 'Значення'

# Обчислення частот та відносних частот
for i in uni_keys:
    df.iloc[uni_keys.index(i)][0] = n[i]
    df.iloc[uni_keys.index(i)][2] = df.iloc[uni_keys.index(i)][0] / len(data)

# Обчислення кумулятивних частот та відносних кумулятивних частот
df.iloc[0][1] = df.iloc[0][0]
df.iloc[0][3] = df.iloc[0][2]
for i in range(1, len(uni_keys)):
    df.iloc[i][1] = df.iloc[i][0] + df.iloc[i - 1][1]
    df.iloc[i][3] = df.iloc[i][2] + df.iloc[i - 1][3]

# Вивід аналізу статистичних характеристик
print(df)
plt.show()

# Обчислення основних статистичних характеристик
N = len(data)
Ex = np.sum(data) / N  # Середнє значення
Dx = np.sum((data - Ex) ** 2) / N  # Дисперсія
Dx_ = Dx * (N / (N - 1))  # Скоригована дисперсія
Median = (data[49] + data[49 + 1]) / 2  # Медіана

# Мода
df['Частоти'] = pd.to_numeric(df['Частоти'])
Mod = df['Частоти'].idxmax()

# Асиметрія та ексцес
asy = (np.sum((data - Ex) ** 3) / N) / ((np.sum((data - Ex) ** 2) / N) ** (3 / 2))
excess = ((np.sum((data - Ex) ** 4) / N) / (Dx ** 2)) - 3

# Вивід основних статистичних характеристик
print('----------')
print('Середнє значення:', Ex)
print('Дисперсія:', Dx)
print('Скоригована дисперсія:', Dx_)
print('Медіана:', Median)
print('Мода:', Mod)
print('Асиметрія:', asy)
print('Ексцес:', excess)
print('----------')

# Мінімальне та максимальне значення
min_el = np.min(uni_keys)
max_el = np.max(uni_keys)
xdistr = np.arange(min_el, max_el + 1)

# Графічний вивід розподілу біноміального закону
def drawTable(nfig, cell_text, labels, title):
    plt.figure(nfig)
    fig = plt.gcf()
    fig.set_size_inches(13, 5)
    plt.axis('off')
    cell_text = np.array([cell_text])
    table = plt.table(cellText=np.around(cell_text, 4), rowLabels=['P'], colLabels=labels,
                      loc='upper center', cellLoc='center')
    table.add_cell(0, -1, table[0, 1].get_width(), table[0, 1].get_height(), text='ξ')
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1, 3.5)
    plt.title(title)

# Біноміальний розподіл
def BinDistribution(x, p, n):
    return scipy.special.binom(n, x) * (p ** x) * (1 - p) ** (n - x)

n_bin = df.index.max()
p_bin = Ex / n_bin
Dbin = n_bin * p_bin * (1 - p_bin)
ybin = BinDistribution(xdistr, p_bin, n_bin)
bindf = pd.DataFrame(index=xdistr, columns=['P'])
bindf['P'] = np.around(ybin, 4)
bindf.index.name = 'ξ'
print('Біноміальний розподіл\n', bindf.T)
print('Сума біноміального розподілу:', np.sum(ybin))
plt.figure(0)
plt.plot(xdistr, ybin)
plt.scatter(xdistr, ybin)
plt.title('Біноміальний розподіл з параметрами n=' + str(n_bin) + '; p=' + str(np.around(p_bin, 4)))
plt.xticks(xdistr)

title = 'Ряд розподiлу Bin(n=' + str(n_bin) + ', p=' + str
