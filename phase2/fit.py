from core import *
import pandas as pd, joblib
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

table=pd.read_csv(OUT/'dataset.csv'); seq=table.sequence.tolist()
x=Features()(seq); np.save(OUT/'dataset_features.npy',x)
y=table.target_log10_uM.to_numpy()
masks={s:table.split.eq(s).to_numpy() for s in ['train','validation','calibration','test']}
models={'ridge10':make_pipeline(StandardScaler(),Ridge(alpha=10)),
 'ridge100':make_pipeline(StandardScaler(),Ridge(alpha=100)),
 'extra_trees':ExtraTreesRegressor(n_estimators=300,min_samples_leaf=3,max_features=0.7,random_state=42,n_jobs=4),
 'random_forest':RandomForestRegressor(n_estimators=300,min_samples_leaf=3,max_features=0.7,random_state=42,n_jobs=4)}
selection={}
for name,model in models.items():
    model.fit(x[masks['train']],y[masks['train']])
    selection[name]=float(mean_absolute_error(y[masks['validation']],model.predict(x[masks['validation']])))
    print(name,selection[name],flush=True)
name=min(selection,key=selection.get); model=models[name]
# Refit only development data; calibration/test labels never influence model choice.
development=masks['train']|masks['validation']; model.fit(x[development],y[development])
joblib.dump(model,OUT/'mic_regressor.joblib')
pred=model.predict(x)
q=conformal_quantile(abs(y[masks['calibration']]-pred[masks['calibration']]))
qu=conformal_quantile(y[masks['calibration']]-pred[masks['calibration']])
table['prediction_log10_uM']=pred; table['lower_log10_uM']=pred-q; table['upper_log10_uM']=pred+q
table['one_sided_upper_log10_uM']=pred+qu
table.to_csv(OUT/'heldout_predictions.csv',index=False)
test=masks['test']; coverage=float(np.mean(abs(y[test]-pred[test])<=q))
report={'selected_model':name,'validation_mae':selection,'alpha':0.1,'calibration_n':int(masks['calibration'].sum()),
 'q_two_sided':q,'q_one_sided':qu,'test_n':int(test.sum()),'test_mae_log10':float(mean_absolute_error(y[test],pred[test])),
 'test_rmse_log10':float(np.sqrt(mean_squared_error(y[test],pred[test]))),'test_spearman':float(spearmanr(y[test],pred[test]).statistic),
 'test_empirical_two_sided_coverage':coverage,'test_empirical_one_sided_coverage':float(np.mean(y[test]<=pred[test]+qu)),
 'test_median_baseline_mae':float(mean_absolute_error(y[test],np.full(test.sum(),np.median(y[development])))),
 'dataset_sha256':sha(OUT/'dataset.csv'),'model_sha256':sha(OUT/'mic_regressor.joblib'),
 'features':'ESM2-8M layer6 residue mean (320) + 10 descriptors (length, net-charge/residue proxy, KD mean, alpha-helical hydrophobic moment, hydrophobic/aromatic/C/P/G fractions, consecutive repeats)',
 'scope':'Exploratory marginal interval for aggregated historical E. coli endpoint. No selected-candidate or hit-rate guarantee.'}
dump(OUT/'calibration.json',report); print(json.dumps(report,indent=2))
