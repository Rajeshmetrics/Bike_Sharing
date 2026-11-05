"""Streamlit natural-language operations assistant. Run: streamlit run app.py"""
from pathlib import Path
import json,re
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parent
results=json.loads((ROOT/'outputs/results.json').read_text())
pred=pd.read_csv(ROOT/'outputs/test_predictions.csv',parse_dates=['timestamp'])
st.set_page_config(page_title='Bike Operations Copilot',layout='wide')
st.title('Bike Operations Copilot')
st.caption('Rule-based natural-language prototype grounded in historical holdout predictions; not a live dispatch system.')
st.info('Scope: citywide demand only. Station-level shortages and actual fleet inventory cannot be inferred from this dataset.')
q=st.text_input('Ask an operations question',placeholder='What is the busiest hour?')
examples=['What is the busiest hour?','How accurate is the model?','How many bikes do we need for a 30 minute trip?','What is the impact of working-day rush hours?','Show forecast for 2012-11-15','Should we buy more bikes?']
st.caption('Try: '+' | '.join(examples))
def answer(question):
 s=question.lower()
 if any(w in s for w in ['accur','error','mae','model','performance']):
  m=results['final_refit_test_metrics'];return f"The selected {results['selected_model']} model achieves MAE {m['MAE']:.1f} rentals/hour, RMSE {m['RMSE']:.1f}, and R² {m['R2']:.3f} on Nov–Dec 2012 held-out data. This uses historical observed weather."
 if 'rush' in s or 'commut' in s or 'working' in s:
  h=results['hypothesis'];return f"Working-day peak hours (07–09 and 17–19) average {h['peak_hour_mean']:.1f} rentals/hour, versus {h['other_hour_mean']:.1f} at other hours. Day-level bootstrap 95% CI for mean within-day difference: {h['day_block_bootstrap_ci95'][0]:.1f}–{h['day_block_bootstrap_ci95'][1]:.1f}. This is association, not causation."
 if 'busiest' in s or 'peak hour' in s:
  raw=pd.read_csv(ROOT/'hour.csv');means=raw.groupby('hr')['cnt'].mean();return f'Historically, hour {means.idxmax():02d}:00 has the highest mean rentals ({means.max():.1f} rentals/hour). This is citywide.'
 if any(w in s for w in ['bike','fleet','buy','purchase','need']):
  match=re.search(r'(15|30|45)\s*(?:min|minute)',s);mins=int(match.group(1)) if match else 30
  scenarios=[x for x in results['fleet_scenarios'] if x['assumed_avg_trip_minutes']==mins]
  a,b=scenarios
  return f"Illustrative scenario at 95th percentile forecast demand ({results['predicted_test_p95_hourly_rentals']:.0f} trip starts/hour), assuming {mins}-minute average trip duration: {a['estimated_citywide_active_bikes_p95']} simultaneously active bikes, or about {a['illustrative_fleet_required']} bikes at 70% assumed utilization / {b['illustrative_fleet_required']} at 85%. These are NOT measured fleet needs. First collect station inventory, unmet demand, trip duration, and rebalancing data."
 date=re.search(r'2012-\d\d-\d\d',s)
 if date:
  day=pred[pred.timestamp.dt.strftime('%Y-%m-%d')==date.group()]
  if day.empty:return 'No test-set forecast available for that date. The holdout spans November–December 2012.'
  peak=day.loc[day.predicted_rentals.idxmax()]
  return f"For {date.group()}, predicted {day.predicted_rentals.sum():.0f} total trip starts across recorded hours; predicted busiest hour {int(peak.hr):02d}:00 with {peak.predicted_rentals:.0f} rentals. These are backtest predictions, not live forecasts."
 return 'I can answer model accuracy, busiest hour, commute-hour effect, fleet scenarios, buying recommendation, and date-specific November–December 2012 backtest forecasts. No external LLM or API key is needed.'
if q:st.markdown('### Answer');st.write(answer(q))
st.subheader('Backtest predictions')
st.line_chart(pred.set_index('timestamp')[['cnt','predicted_rentals']].iloc[:240])
st.dataframe(pred.head(48),use_container_width=True)
