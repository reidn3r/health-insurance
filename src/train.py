# %% [markdown]
# # Health Insurance Cross-Sell: Predição de Propensão
#
# - Objetivo: identificar clientes com maior propensão a contratar seguro de veículo, priorizando o contato da equipe de vendas.
# - Tratado como ranking por probabilidade, sem threshold fixo. Métrica principal: ROC-AUC, complementada por precision@k e recall@k.
# %%
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("/home/reidner/dev/portfolio/ds-health-insurer/data/train.csv")

print(f'train shape: {df.shape}')
print(f'train nan:\n{df.isna().sum()}')
print(f'train cols: {df.columns}')

df.head()
# %%
print('region dtype: ', df['Region_Code'].dtype)
print('region values: ', df['Region_Code'].unique())

df['Region_Code'] = df['Region_Code'].astype(int).astype(str)
df['Region_Code'].head()
# %%
print('policy slaes channel dtype: ', df['Policy_Sales_Channel'].dtype)
print('sales channel values: ', df['Policy_Sales_Channel'].unique())

df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype(int).astype(str)
df['Policy_Sales_Channel'].head()
# %%
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer
from feature_engine.encoding import CountFrequencyEncoder

df['prev_and_damage'] = (
  (df['Previously_Insured'] == 1) & (df['Vehicle_Damage'] == 'Yes')
  ).astype(int)
df['log_annual_premium'] = np.log1p(df['Annual_Premium'])

print(f'prev_and_damage rate: {df["prev_and_damage"].mean():.4f}')
print(f'log_annual_premium range: {df["log_annual_premium"].min():.2f} - {df["log_annual_premium"].max():.2f}')

ages = df['Vehicle_Age'].unique()[::-1]
print(f'unique ages: {ages}')

# ordinal_encoder = OrdinalEncoder(categories=[ages], dtype=np.uint8)
# df['age_encoded'] = ordinal_encoder.fit_transform(df[['Vehicle_Age']])

# ohe_encoder = OneHotEncoder()
# damage_encoded = ohe_encoder.fit_transform(df[['Vehicle_Damage']])
# damage_cols = ohe_encoder.get_feature_names_out(['Vehicle_Damage'])
# print('damage cols:', damage_cols)

x = df.copy()

preprocessor = ColumnTransformer(
  transformers=[
    ('vehicle_age', OrdinalEncoder(categories=[ages], dtype=np.uint8), ['Vehicle_Age']),
    ('vehicle_damage', OneHotEncoder(), ['Vehicle_Damage']),
    ('gender_encoded', OneHotEncoder(), ['Gender']),
    ('age_binner', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile'), ['Age']),
    ('region_frequency', CountFrequencyEncoder(encoding_method='count'), ['Region_Code']),
    ('channel_frequency', CountFrequencyEncoder(encoding_method='count'), ['Policy_Sales_Channel']),
  ],
  remainder='passthrough',
  verbose_feature_names_out=False
)
from sklearn.model_selection import train_test_split

X_raw = df.drop(['id', 'Response'], axis=1)
y = df['Response']

xtrain, xtest, ytrain, ytest = train_test_split(X_raw, y, test_size=0.2, stratify=y, random_state=42)
print(f'train/test target ratio: {np.mean(ytrain):.4f}, {np.mean(ytest):.4f}')
print(f'train/test shape (raw): {xtrain.shape}, {xtest.shape}')
# %% [markdown]
# - precision_at_k e recall_at_k medem a decisão negócio (contatar só o top k% do ranking).
# - Complementam o AUC: ele é agregado sobre todos os thresholds e não mostra quão eficiente/completo é esse recorte.
# %%
# Utilitárias
def precision_at_k(y_true, y_score, k):
    # Fração de positivos reais entre os k% melhores ranqueados.
    y_true = np.asarray(y_true)
    n = int(np.ceil(k * len(y_score)))
    top = np.argsort(y_score)[::-1][:n]
    return y_true[top].mean()

def recall_at_k(y_true, y_score, k):
    # Fração dos positivos reais capturados nos k% melhores ranqueados.
    y_true = np.asarray(y_true)
    n = int(np.ceil(k * len(y_score)))
    top = np.argsort(y_score)[::-1][:n]
    return y_true[top].sum() / y_true.sum()

# %% [markdown]
# DummyClassifier: piso de comparação, com AUC aproximadamente 0.5 esperado, para verificar se os modelos seguintes aprendem a relacionar as co-variáveis com a variável objetivo.
# %%
import mlflow
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score

METRIC_COLUMN = 'auc' #p/ padronizar no mlflow
with mlflow.start_run(run_name="baseline-dummy"):
    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(xtrain, ytrain)
    yhat_dummy = dummy.predict_proba(xtest)[:, 1]
    auc_dummy = roc_auc_score(ytest, yhat_dummy)
    print(f'dummy roc-auc: {auc_dummy}')
    mlflow.log_metric(METRIC_COLUMN, auc_dummy)


# %% [markdown]
# Decision Tree como 2º baseline: mais simples que ensemble, rápida e interpretável.
# Serve de piso "razoável" antes de modelos custosos e permite inspecionar a árvore e as importâncias.
# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("health-cross-sell")

baseline_params = {
    "class_weight": None,
    # "class_weight": "balanced",
    "max_depth": 15,
    "min_samples_leaf": 10,
    "random_state": 42,
}

with mlflow.start_run(run_name="baseline-dt"):
    print(f'starting baseline dt train')
    mlflow.log_params(baseline_params)
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', DecisionTreeClassifier(**baseline_params)),
    ])

    model.fit(xtrain, ytrain)
    signature = infer_signature(xtrain.head(100), model.predict_proba(xtrain.head(100)))
    mlflow.sklearn.log_model(
        model,
        name="model",
        signature=signature,
        registered_model_name="health-cross-sell-baseline-dt-classifier",
        serialization_format="cloudpickle",
    )

    yhat = model.predict_proba(xtest)[:, 1]
    auc = roc_auc_score(ytest, yhat)

    print(f'baseline roc-auc: {auc}')
    mlflow.log_metric(METRIC_COLUMN, auc)

    precision = precision_at_k(ytest, yhat, 0.05)
    print(f'baseline precision@5: {precision}')
    mlflow.log_metric("precision_at_5pct", precision)

    recall = recall_at_k(ytest, yhat, 0.05)
    print(f'baseline recall@5: {recall}')
    mlflow.log_metric("recall_at_5pct", recall)

# %%
from sklearn.tree import plot_tree

feature_names = model.named_steps['preprocessor'].get_feature_names_out().tolist()
dt = model.named_steps['clf']

plt.figure(figsize=(25, 12))
plot_tree(
    dt,
    feature_names=feature_names,
    max_depth=5,
    filled=True,
    fontsize=10,
)
plt.title('Decision Tree')
plt.show()
# %%
importances = pd.Series(dt.feature_importances_, index=feature_names)
baseline_sum = (importances
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature", 0:"importance"})
       )
baseline_sum['sum'] = baseline_sum["importance"].cumsum()
baseline_sum
# %% [markdown]
# - Leitura das importâncias: Vehicle_Damage_Yes domina, seguida de Age, Vintage e Policy_Sales_Channel.
# %% [markdown]
# Random Forest: próximo candidato, com tuning bayesiano via Optuna.
# Validação cruzada estratificada e média das AUCs no objective (em vez de um split único) torna a busca robusta à variância de amostragem.
# %%
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.model_selection import cross_validate

def rf_objective(trial: optuna.Trial):
  with mlflow.start_run(nested=True, run_name=f"rf_trial_{trial.number}") as child:
    params = {
      "class_weight": None,
      "max_depth": trial.suggest_int("rf_max_depth", 5, 25),
      "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 5, 200),
      "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 50),
      "max_features": trial.suggest_categorical("rf_max_features", ["sqrt", "log2"]),
      "n_estimators": trial.suggest_int("rf_n_estimators", 100, 500),
      "random_state": 42,
      "n_jobs": 3
    }

    mlflow.log_params(params)
    model = Pipeline([
      ("preprocessor", preprocessor),
      ("classifier", RandomForestClassifier(**params))
    ])

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_results = cross_validate(
      model, xtrain, ytrain,
      cv=cv, scoring="roc_auc",
      n_jobs=1,
      return_estimator=True,
    )
    scores = cv_results["test_score"]
    auc_mean = scores.mean()
    auc_std = scores.std()

    mlflow.log_metric(METRIC_COLUMN, auc_mean)
    mlflow.log_metric(f"{METRIC_COLUMN}_std", auc_std)

    fitted_model = cv_results["estimator"][0]
    signature = infer_signature(xtrain.head(100), fitted_model.predict_proba(xtrain.head(100)))
    mlflow.sklearn.log_model(
      fitted_model,
      name="model",
      signature=signature,
      serialization_format="cloudpickle",
      registered_model_name=f"health-cross-sell-rf-classifier-{trial.number}",
    )

    trial.set_user_attr("run_id", child.info.run_id)
    return auc_mean
    
with mlflow.start_run(run_name="rf-study") as run:
    n_trials = 5
    mlflow.log_param("n_trials", n_trials)

    study = optuna.create_study(direction="maximize")
    study.optimize(rf_objective, n_trials=n_trials)

    # mlflow.log_params(study.best_params)
    mlflow.log_params(study.best_trial.params)
    mlflow.log_metric("best_auc", study.best_value)

    if best_run_id := study.best_trial.user_attrs.get("run_id"):
        mlflow.log_param("best_child_run_id", best_run_id)

# %%
best_pipeline = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
yhat = best_pipeline.predict_proba(xtest)[:, 1]

precision = precision_at_k(ytest, yhat, 0.05)
print(f'rf precision@5: {precision}')
mlflow.log_metric("precision_at_5pct", precision)

recall = recall_at_k(ytest, yhat, 0.05)
print(f'rf recall@5: {recall}')
mlflow.log_metric("recall_at_5pct", recall)
# %%
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(ytest, yhat, n_bins=10, strategy="quantile")
plt.figure(figsize=(7, 6))
plt.plot(prob_pred, prob_true, marker="o")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Calibração perfeita")
plt.xlabel("Probabilidade predita média")
plt.ylabel("Fração real de positivos")
plt.title("Calibration Curve — melhor RF")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
# %% [markdown]
# Random forest: precision@5% {0.399}, recall@5% {0.1625}.
# Calibração comprimida, típico de bagging e média de árvores: scores sem extremos. Não prejudica o ranking, apenas a leitura da probabilidade como confiança calibrada.
# %%
from catboost import CatBoostClassifier

mlflow.end_run()
def cb_objective(trial: optuna.Trial):
  with mlflow.start_run(nested=True, run_name=f"cb_trial_{trial.number}") as child:
    params = {
      "iterations": trial.suggest_int("cb_iterations", 200, 800),
      "depth": trial.suggest_int("cb_depth", 4, 10),
      "learning_rate": trial.suggest_float("cb_learning_rate", 0.01, 0.3, log=True),
      "l2_leaf_reg": trial.suggest_float("cb_l2_leaf_reg", 1.0, 10.0, log=True),
      "bagging_temperature": trial.suggest_float("cb_bagging_temperature", 0.0, 1.0),
      "random_strength": trial.suggest_float("cb_random_strength", 0.0, 10.0),
      "border_count": trial.suggest_int("cb_border_count", 32, 255),
      "random_state": 42,
      "verbose": False,
      "allow_writing_files": False,
    }

    mlflow.log_params(params)
    model = Pipeline([
      ("preprocessor", preprocessor),
      ("classifier", CatBoostClassifier(**params))
    ])

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_results = cross_validate(
      model, xtrain, ytrain,
      cv=cv, scoring="roc_auc",
      n_jobs=1,
      return_estimator=True,
    )
    scores = cv_results["test_score"]
    auc_mean = scores.mean()
    auc_std = scores.std()

    mlflow.log_metric(METRIC_COLUMN, auc_mean)
    mlflow.log_metric(f"{METRIC_COLUMN}_std", auc_std)

    fitted_model = cv_results["estimator"][0] 
    signature = infer_signature(xtrain.head(100), fitted_model.predict_proba(xtrain.head(100)))
    mlflow.sklearn.log_model(
      fitted_model,
      name="model",
      signature=signature,
      serialization_format="cloudpickle",
      registered_model_name=f"health-cross-sell-cb-classifier-{trial.number}",
    )

    trial.set_user_attr("run_id", child.info.run_id)
    return auc_mean

with mlflow.start_run(run_name="cb-study") as run:
    n_trials = 5
    mlflow.log_param("n_trials", n_trials)
    mlflow.set_tag("stage", "tuning")

    study = optuna.create_study(direction="maximize")
    study.optimize(cb_objective, n_trials=n_trials)

    mlflow.log_params(study.best_trial.params)
    mlflow.log_metric("best_auc", study.best_value)

    if best_run_id_cb := study.best_trial.user_attrs.get("run_id"):
        mlflow.log_param("best_child_run_id", best_run_id_cb)

# %%
best_pipeline_cb = mlflow.sklearn.load_model(f"runs:/{best_run_id_cb}/model")
yhat_cb = best_pipeline_cb.predict_proba(xtest)[:, 1]

precision = precision_at_k(ytest, yhat_cb, 0.05)
print(f'rf precision@5: {precision}')
mlflow.log_metric("precision_at_5pct", precision)

recall = recall_at_k(ytest, yhat_cb, 0.05)
print(f'rf recall@5: {recall}')
mlflow.log_metric("recall_at_5pct", recall)
# %%
prob_true_cb, prob_pred_cb = calibration_curve(ytest, yhat_cb, n_bins=10, strategy="quantile")

plt.figure(figsize=(7, 6))
plt.plot(prob_pred_cb, prob_true_cb, marker="o", label="CatBoost (best trial)")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Calibração perfeita")
plt.xlabel("Probabilidade predita média")
plt.ylabel("Fração real de positivos")
plt.title("Calibration Curve — melhor CatBoost")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
# %% [markdown]
## Escolha do modelo final: CatBoost
# - Depois de comparar Decision Tree (baseline), Random Forest e CatBoost com tuning via Optuna e validação cruzada estratificada, o CatBoost foi escolhido como modelo final.
# - Em termos de AUC os dois principais candidatos ficaram próximos. CatBoost atingiu uma AUC média de validação cruzada de 0.8543 contra 0.8510 do Random Forest, uma diferença pequena mas consistente a favor do CatBoost. Precision e recall no topo do ranking, considerando top 5% e top 10%, também ficaram no mesmo patamar entre os dois modelos.
# - Em relação à qualidade da calibração das probabilidades, CatBoost e Random Forest apresentaram curvas muito semelhantes. Ambos ficam bem próximos da diagonal ideal na faixa de probabilidades observada, e ambos concentram os scores numa faixa relativamente estreita (até aproximadamente 0.4), sem produzir probabilidades extremas próximas de 0 ou 1. Esse comportamento é típico de modelos baseados em ensembles de árvores. Como não há diferença relevante de calibração entre os dois candidatos, esse critério não foi determinante na escolha do modelo final.

# %% [markdown]
# Retreino final com todos os dados, usando hiperparâmetros já validados por CV.
# A partir daqui, xtest deixa de ser hold-out válido p/ este modelo (foi incorporado ao treino).
# %%
from sklearn.preprocessing import FunctionTransformer

def cast_categoricals(df):
    df = df.copy()
    df['Region_Code'] = df['Region_Code'].astype(int).astype(str)
    df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype(int).astype(str)
    return df

final_params = {**study.best_trial.params, "random_state": 42, "verbose": False, "allow_writing_files": False}
final_params = {k.replace("cb_", ""): v for k, v in final_params.items()}

catboost = Pipeline([
    ("cast", FunctionTransformer(cast_categoricals)),
    ("preprocessor", preprocessor),
    ("classifier", CatBoostClassifier(**final_params))
])

catboost.fit(X_raw, y)
signature = infer_signature(X_raw.head(100), catboost.predict_proba(X_raw.head(100)))

mlflow.end_run()
with mlflow.start_run(run_name="final-catboost"):
  mlflow.set_tag("stage", "production")
  mlflow.log_params(final_params)
  mlflow.sklearn.log_model(
    catboost,
    name="final-model",
    signature=signature,
    serialization_format="cloudpickle",
    registered_model_name="health-cross-sell-final",
  )

# %%
from sklearn.metrics import roc_curve, auc

yhat = catboost.predict_proba(xtest)[:, 1]
fpr, tpr, thresholds = roc_curve(ytest, yhat)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="C0", label=f"CatBoost (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Aleatório (AUC = 0.5)")
plt.xlabel("Taxa de Falsos Positivos (FPR)")
plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %%
df_test = pd.read_csv("/home/reidner/dev/portfolio/ds-health-insurer/data/test.csv")
df_test['prev_and_damage'] = (
    (df_test['Previously_Insured'] == 1) & (df_test['Vehicle_Damage'] == 'Yes')
).astype(int)
df_test['log_annual_premium'] = np.log1p(df_test['Annual_Premium'])

X_test = df_test.drop(['id'], axis=1)

scores = catboost.predict_proba(X_test)[:, 1]
ranking = pd.DataFrame({
    'id': df_test['id'],
    'Response_score': scores
}).sort_values('Response_score', ascending=False)

ranking.iloc[:15, :]
# %% [markdown]
# Conclusão
# - Modelo final: CatBoost, AUC 0.8652 obtida como média de validação cruzada durante o tuning.
# - Features de destaque, segundo análise da árvore de decisão: Vehicle_Damage, Age, Vintage.
# - Limitações: dataset desbalanceado, cerca de 12% de positivos e AUC observada neste dataset na faixa de 0.85 a 0.86 com difícil progressão.

# %%
