# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
def jupyter_settings():
    from IPython.display import display, HTML

    sns.set_theme(context="notebook", style="whitegrid")
    plt.rcParams['figure.figsize'] = [25, 12]
    plt.rcParams['font.size'] = 24
    plt.rcParams['figure.dpi'] = 110
    plt.rcParams['savefig.dpi'] = 110
    plt.rcParams['figure.facecolor'] = 'white'

    # Pandas display options
    display(HTML('<style>.container { width:100% !important; }</style>'))
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.set_option('display.expand_frame_repr', False)

jupyter_settings()
# %%
train = pd.read_csv("/home/reidner/dev/portfolio/ds-health-insurer/data/train.csv")

print(f'train shape: {train.shape}')
print(f'train nan:\n{train.isna().sum()}')
print(f'train cols: {train.columns}')
#Region code de onde? podemos pegar zonas/estados/localizações

train.head()
# %%
train.info()
# %%
train.describe()
# %% [markdown]
# ## Distribuição das variáveis categóricas
# %% [markdown]
# * Cerca de 88% dos clientes são Response=0. 
# * Avaliar métrica de um modelo por acurácia é enganoso, avaliar por AUC/ROC.
# %%
fig, axes = plt.subplots(1, 3, figsize=(25, 12))
for ax, col in zip(axes, ['Gender', 'Driving_License', 'Response']):
    counts = train[col].value_counts(normalize=True)
    counts.plot(kind='bar', ax=ax, width=0.6)
    ax.set_title(col)
    ax.set_ylabel('Proporção')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.bar_label(ax.containers[0], fmt='%.0f%%')

plt.tight_layout()
plt.show()

print(f'Taxa de Response=1: {train["Response"].mean()}')
# %% [markdown]
# ## Distribuição das variáveis numéricas por Response
exclude_cols = ["id", "Region_Code", "Response"]
cols = [c for c in train.columns if train[c].dtype != "object" and c not in exclude_cols]
ncols, nrows = 3, 2
fig, axes = plt.subplots(nrows, ncols, figsize=(25, 12))
axes_flat = axes.flatten()

for ax, col in zip(axes_flat, cols):
    sns.kdeplot(data=train, x=col, hue="Response", fill=True, alpha=0.5, ax=ax)

for ax in axes_flat[len(cols):]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()
# %%
sns.countplot(data=train, x="Previously_Insured", hue="Response")
plt.title("Previously_Insured x Response")
plt.xticks(rotation=0)
plt.show()
# %% [markdown]
# Quem já tem seguro de veículo (1) nunca contrata novo seguro, logo, feature altamente discriminante
# %%
fig, axes = plt.subplots(1, 2, figsize=(25, 12))
for ax, col in zip(axes, ['Vehicle_Age', 'Vehicle_Damage']):
    counts = train[col].value_counts(normalize=True)
    counts.plot(kind='bar', ax=ax, width=0.6)
    ax.set_title(col)
    ax.set_ylabel('Proporção')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.bar_label(ax.containers[0], fmt='%.0f%%')

plt.tight_layout()
plt.show()
# %% [markdown]
# ## Correlação entre variáveis numéricas
# %% [markdown]
# * multicolinearidade entre features pode indicar redundância de informação
# * correlação alta com Response indica poder preditivo.
# %%
num_cols = [c for c in train.columns if train[c].dtype != "object" and c not in ["id", "Region_Code"]]
plt.figure(figsize=(25, 12))
sns.heatmap(train[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title("Matriz de Correlação")
plt.show()
# %% [markdown]
# ## Boxplots por Response
# %% [markdown]
# Compara mediana e outliers entre quem contrata (1) e quem não contrata.
# %%
box_cols = ['Age', 'Annual_Premium', 'Vintage', 'Policy_Sales_Channel']
ncols = 2
nrows = int(np.ceil(len(box_cols) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(25, 12))
axes_flat = axes.flatten()

for ax, col in zip(axes_flat, box_cols):
    sns.boxplot(data=train, x="Response", y=col, ax=ax)
    ax.set_title(col)

plt.tight_layout()
plt.show()
# %% [markdown]
# ## Annual_Premium alto x Response
# %%
for q in [0.5, 0.90, 0.99]:
    cutoff = train['Annual_Premium'].quantile(q)
    hi = train[train['Annual_Premium'] >= cutoff]
    low = train[train['Annual_Premium'] < cutoff]
    print(f'p{int(q*100)}={cutoff:.0f} | n hi: {len(hi):,} ({len(hi)/len(train):.1%}) | resp hi: {hi.Response.mean():.3f} | resp low: {low.Response.mean():.3f}')

fig, axes = plt.subplots(1, 3, figsize=(25, 12))
for ax, q in zip(axes, [0.5, 0.90, 0.99]):
    cutoff = train['Annual_Premium'].quantile(q)
    hi = train[train['Annual_Premium'] >= cutoff]
    sns.kdeplot(data=hi, x='Annual_Premium', hue='Response', fill=True, alpha=0.5, ax=ax)
    ax.set_title(f'Annual_Premium >= p{int(q*100)} ({cutoff:.0f})')
plt.tight_layout()
plt.show()
# %% [markdown]
# ## Possíveis novas variáveis
#
# - Vehicle_Damage vira binária 
# - Vehicle_Age já é ordenada: virar ordinal (0, 1, 2) em vez de 3 dummies 
# - Age em faixas (ex.: 5 bins) 
# - region_freq: frequência normalizada do Region_Code  Precisa ser calculada só em train
# - channel_freq: o mesmo para Policy_Sales_Channel (155 níveis)
# Dados geográficos externos ficaram de fora: o Region_Code é anonimizado 