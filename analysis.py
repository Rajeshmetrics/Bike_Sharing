"""SIGMOID bike-sharing case study. Run: python analysis.py"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
import joblib

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'; FIG=ROOT/'figures'
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
H=pd.read_csv(ROOT/'hour.csv'); D=pd.read_csv(ROOT/'day.csv')
H['dteday']=pd.to_datetime(H['dteday']);D['dteday']=pd.to_datetime(D['dteday'])
H['timestamp']=H['dteday']+pd.to_timedelta(H['hr'],unit='h')
H=H.sort_values('timestamp').reset_index(drop=True)
checks={
'hour_rows':len(H),'day_rows':len(D),'hour_null_cells':int(H.isna().sum().sum()),'day_null_cells':int(D.isna().sum().sum()),
'hour_duplicate_timestamps':int(H['timestamp'].duplicated().sum()),'day_duplicate_dates':int(D['dteday'].duplicated().sum()),
'hour_cnt_identity_failures':int((H['casual']+H['registered']!=H['cnt']).sum()),
'day_cnt_identity_failures':int((D['casual']+D['registered']!=D['cnt']).sum()),
'negative_hour_counts':int((H[['cnt','casual','registered']]<0).sum().sum()),
'daily_hourly_total_mismatch_days':int((H.groupby('dteday')['cnt'].sum().reindex(D['dteday']).to_numpy()!=D['cnt'].to_numpy()).sum()),
'missing_hour_slots':int(len(pd.date_range(H['timestamp'].min(),H['timestamp'].max(),freq='h'))-len(H)),
'date_start':str(H['dteday'].min().date()),'date_end':str(H['dteday'].max().date()),
}
# Some hours may be absent; do not fill missing rentals with zero.
H['is_commute_peak']=H['hr'].isin([7,8,9,17,18,19]).astype(int)
H['is_weekend']=(H['workingday']==0).astype(int) # non-working day, including holidays
H['temp_c']=H['temp']*41
H['humidity_pct']=H['hum']*100
H['month_sin']=np.sin(2*np.pi*H['mnth']/12)
H['month_cos']=np.cos(2*np.pi*H['mnth']/12)
H['hour_sin']=np.sin(2*np.pi*H['hr']/24)
H['hour_cos']=np.cos(2*np.pi*H['hr']/24)
# Hypothesis: On working days, commute-peak hours have higher rentals than other hours.
a=H.loc[(H.workingday==1)&(H.is_commute_peak==1),'cnt'].to_numpy()
b=H.loc[(H.workingday==1)&(H.is_commute_peak==0),'cnt'].to_numpy()
tstat,pval=ttest_ind(a,b,equal_var=False)
# Day-block bootstrap: preserve within-day dependence for difference of mean hourly demand.
wd=H[H.workingday==1].copy()
daily=wd.groupby(['dteday','is_commute_peak'])['cnt'].mean().unstack().dropna()
differences=(daily[1]-daily[0]).to_numpy()
rng=np.random.default_rng(42)
boot=np.mean(rng.choice(differences,size=(3000,len(differences)),replace=True),axis=1)
hyp={'hypothesis':'Working-day commute peak (07-09, 17-19) hourly rental demand exceeds other working-day hours',
'peak_hour_mean':float(a.mean()),'other_hour_mean':float(b.mean()),'difference':float(a.mean()-b.mean()),
'welch_t_stat':float(tstat),'welch_p_value':float(pval),'day_block_bootstrap_mean_difference':float(differences.mean()),
'day_block_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),'working_days_with_both_groups':len(differences),
'caveat':'Observational association; Welch test ignores within-day dependence, so day-level bootstrap is preferred for uncertainty. Not a causal effect.'}
# Time-ordered holdout. Train 2011 through Sep 2012; validate Oct 2012; test Nov-Dec 2012.
train=H[H.dteday<'2012-10-01'];valid=H[(H.dteday>='2012-10-01')&(H.dteday<'2012-11-01')];test=H[H.dteday>='2012-11-01']
FEATURES=['season','yr','mnth','hr','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed','is_commute_peak','month_sin','month_cos','hour_sin','hour_cos']
# Never use casual, registered, cnt, instant, timestamp or future observations as predictors.
models={
'mean_baseline':DummyRegressor(strategy='mean'),
'random_forest':RandomForestRegressor(n_estimators=160,min_samples_leaf=3,max_features=.85,n_jobs=-1,random_state=42),
'hist_gradient_boosting':HistGradientBoostingRegressor(max_iter=220,learning_rate=.06,max_leaf_nodes=24,l2_regularization=4,random_state=42)
}
def metrics(y,p):
 p=np.maximum(0,p)
 return {'MAE':float(mean_absolute_error(y,p)),'RMSE':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p))}
results={}
for name,m in models.items():
 m.fit(train[FEATURES],train.cnt)
 results[name]={'validation':metrics(valid.cnt,m.predict(valid[FEATURES])),'test':metrics(test.cnt,m.predict(test[FEATURES]))}
# Select on validation only, then refit on train+validation, evaluate final untouched test.
best=min(models,key=lambda n:results[n]['validation']['MAE'])
selected=models[best]
trainval=H[H.dteday<'2012-11-01']
selected.fit(trainval[FEATURES],trainval.cnt)
pred=np.maximum(0,selected.predict(test[FEATURES]))
final_test=metrics(test.cnt,pred)
joblib.dump({'model':selected,'features':FEATURES,'model_name':best},OUT/'demand_model.joblib')
forecast=test[['timestamp','dteday','hr','cnt','workingday','weathersit']].copy()
forecast['predicted_rentals']=np.round(pred,1)
forecast.to_csv(OUT/'test_predictions.csv',index=False)
# Residual uncertainty: calibration on validation from initial trained model (before refit).
# A symmetric conservative additive margin is reported, not a formal conditional prediction interval.
initial=models[best]
# It has been refit, so obtain held-out calibration using a new same-class clone.
from sklearn.base import clone
cal=clone(initial).fit(train[FEATURES],train.cnt)
valid_errors=np.abs(valid.cnt.to_numpy()-np.maximum(0,cal.predict(valid[FEATURES])))
margin=float(np.quantile(valid_errors,.9))
# Operational demand sizing proxy: hourly trip starts -> simultaneous bikes via Little's law,
# under explicitly assumed mean trip duration and uniform arrivals within the hour.
# City-wide only; no station-level shortages, rebalancing or actual fleet utilization identifiable.
peak_hour=float(np.quantile(pred,.95))
scenarios=[]
for minutes in [15,30,45]:
 concurrency=peak_hour*minutes/60
 for target_utilization in [.7,.85]:
  scenarios.append({'assumed_avg_trip_minutes':minutes,'target_fleet_utilization':target_utilization,
   'estimated_citywide_active_bikes_p95':int(np.ceil(concurrency)),
   'illustrative_fleet_required':int(np.ceil(concurrency/target_utilization))})
pd.DataFrame(scenarios).to_csv(OUT/'fleet_scenarios.csv',index=False)
# Figures
plt.figure(figsize=(9,4));H.groupby('hr')['cnt'].mean().plot(kind='bar');plt.title('Average rentals by hour (2011–2012)');plt.ylabel('Rentals / hour');plt.xlabel('Hour');plt.tight_layout();plt.savefig(FIG/'hourly_demand.png',dpi=160);plt.close()
plt.figure(figsize=(9,4));H.groupby('mnth')['cnt'].mean().plot(marker='o');plt.title('Average hourly rentals by month');plt.ylabel('Rentals / hour');plt.xlabel('Month');plt.xticks(range(1,13));plt.tight_layout();plt.savefig(FIG/'monthly_demand.png',dpi=160);plt.close()
plt.figure(figsize=(10,4));plt.plot(test.timestamp,test.cnt,label='Actual',alpha=.65,lw=.9);plt.plot(test.timestamp,pred,label='Predicted',alpha=.7,lw=.9);plt.title('Out-of-time test: Nov–Dec 2012');plt.ylabel('Rentals per hour');plt.legend();plt.tight_layout();plt.savefig(FIG/'actual_vs_predicted.png',dpi=160);plt.close()
summary={'quality':checks,'dataset':{'mean_hourly_rentals':float(H.cnt.mean()),'max_hourly_rentals':int(H.cnt.max()),'total_rentals':int(H.cnt.sum()),'mean_daily_rentals':float(D.cnt.mean()),'train_rows':len(train),'validation_rows':len(valid),'test_rows':len(test)},'hypothesis':hyp,'models':results,'selected_model':best,'final_refit_test_metrics':final_test,'validation_absolute_error_p90':margin,'predicted_test_p95_hourly_rentals':peak_hour,'fleet_scenarios':scenarios,'important_notes':['Demand is completed rentals, not unconstrained requests.','Hourly counts are trip starts, not simultaneously occupied bicycles.','Missing hours are not automatically zero.','No station inventory, trip durations, returns, or purchase costs are supplied.','Forecast is tested on 2012 Nov-Dec; historical weather observed, not a future weather forecast.','The fleet scenarios are illustrative assumptions, not a purchasing recommendation.']}
(OUT/'results.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({'quality':checks,'hypothesis':hyp,'models':results,'selected_model':best,'final_test':final_test,'fleet_scenarios':scenarios},indent=2))
