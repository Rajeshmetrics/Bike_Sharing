from pathlib import Path
import json
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'outputs/results.json').read_text()); q=r['quality'];h=r['hypothesis'];m=r['final_refit_test_metrics']
doc=SimpleDocTemplate(str(ROOT/'CASE_STUDY_REPORT.pdf'),pagesize=(595,842),rightMargin=43,leftMargin=43,topMargin=45,bottomMargin=42)
s=getSampleStyleSheet();s.add(ParagraphStyle(name='T',parent=s['Title'],fontSize=18,leading=22,textColor=colors.HexColor('#16345b'),spaceAfter=10));s.add(ParagraphStyle(name='H',parent=s['Heading2'],fontSize=12,textColor=colors.HexColor('#16345b'),spaceBefore=14,spaceAfter=6));s.add(ParagraphStyle(name='B',parent=s['BodyText'],fontSize=9,leading=13,spaceAfter=7))
story=[]
def p(t):story.append(Paragraph(t,s['B']))
def head(t):story.append(Paragraph(t,s['H']))
def img(name):
 im=Image(str(ROOT/'figures'/name));im.drawWidth=500;im.drawHeight=im.imageHeight*500/im.imageWidth;story.append(im)
def table(rows,widths=None):
 t=Table([[Paragraph(str(x),s['B']) for x in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eef7')),('LINEBELOW',(0,0),(-1,0),.6,colors.grey),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.append(t)
story.append(Paragraph('SIGMOID | Bike-Sharing Case Study',s['T']))
p('Demand forecasting, operational decision support, and fleet-sizing assumptions | Dataset: Capital Bikeshare, Washington D.C., 2011–2012')
head('1. Executive recommendation')
p('<b>Do not approve a blanket fleet purchase based on these data alone.</b> Forecast citywide trip-start demand and target operational availability during commuter peaks. Pilot bike rebalancing and instrument unmet demand at station level before deciding whether to expand the fleet. Historical completed rentals cannot reveal rejected rental requests or the exact number of bikes needed.')
head('2. Data quality and understanding')
p(f"Hourly observations: <b>{q['hour_rows']:,}</b>; daily observations: <b>{q['day_rows']:,}</b>. Date range: {q['date_start']} to {q['date_end']}. Null cells: {q['hour_null_cells']} hourly / {q['day_null_cells']} daily. Duplicate hourly timestamps: {q['hour_duplicate_timestamps']}. Negative rental counts: {q['negative_hour_counts']}. Every row satisfies casual + registered = cnt; hourly and daily totals reconcile. There are <b>{q['missing_hour_slots']}</b> absent hourly timestamps across the full calendar; these were not imputed as zero.")
p('Target: cnt (completed rentals per hour). Inputs: hour, weekday, working-day/holiday flags, season, year/month, weather category, normalized temperature, apparent temperature, humidity, windspeed, and cyclical time features. Exclude casual and registered (direct target leakage), instant (row identifier), and the target itself. The readme documents normalization: temp × 41°C, atemp × 50°C, humidity × 100%, windspeed × 67.')
img('hourly_demand.png')
head('3. Testable hypothesis and experiment')
p('<b>H0:</b> Mean hourly rentals during working-day commute hours (07–09 and 17–19) are no higher than other working-day hours. <b>H1:</b> Commute-hour mean is higher. This is a prespecified comparison, not an intervention.')
p(f"Observed means: <b>{h['peak_hour_mean']:.1f}</b> versus <b>{h['other_hour_mean']:.1f}</b> rentals/hour, raw difference <b>{h['difference']:.1f}</b>. Welch test p-value is smaller than machine reporting precision (report p &lt; 0.001). The more dependence-aware day-block bootstrap yields a 95% CI of <b>{h['day_block_bootstrap_ci95'][0]:.1f} to {h['day_block_bootstrap_ci95'][1]:.1f}</b> rentals/hour for the within-day mean difference over {h['working_days_with_both_groups']} working days. This supports association, not causality; season, special events and commuting patterns may confound it.")
head('4. Forecasting experiment')
p('Chronological train: Jan 2011–Sep 2012; validation: Oct 2012; untouched test: Nov–Dec 2012. Compare mean baseline, random forest, and histogram gradient boosting using MAE. Select only on validation MAE; refit selected model on train + validation and evaluate once on test. Weather features in the backtest are observed, not weather forecasts.')
table([['Model','Validation MAE','Initial test MAE']]+[[name, f"{v['validation']['MAE']:.2f}",f"{v['test']['MAE']:.2f}"] for name,v in r['models'].items()],[215,135,135])
p(f"Selected: <b>{r['selected_model']}</b>. Final refit held-out test MAE <b>{m['MAE']:.2f}</b> rentals/hour, RMSE <b>{m['RMSE']:.2f}</b>, R² <b>{m['R2']:.3f}</b>. This is historical backtest performance, not proof of current-year generalization. The 90th percentile absolute validation error is {r['validation_absolute_error_p90']:.1f} rentals/hour, a rough error buffer rather than a calibrated prediction interval.")
img('actual_vs_predicted.png')
head('5. Estimating bikes needed: explicit scenario assumptions')
p(f"Predicted 95th-percentile hourly trip starts on the test period: <b>{r['predicted_test_p95_hourly_rentals']:.1f}</b>. Proxy active bikes = hourly trip starts × assumed average trip duration (hours). Proxy fleet = active bikes / assumed fleet utilization. This uses an average-rate approximation and ignores within-hour surges, station distribution, maintenance and rebalancing. It is <b>not</b> a purchasing estimate.")
table([['Trip duration','Utilization','Active bikes','Illustrative fleet']]+[[str(x['assumed_avg_trip_minutes'])+' min',f"{x['target_fleet_utilization']:.0%}",str(x['estimated_citywide_active_bikes_p95']),str(x['illustrative_fleet_required'])] for x in r['fleet_scenarios']],[130,110,120,125])
head('6. Operations prototype')
p('The supplied Streamlit application accepts natural-language questions about the busiest hour, test-set accuracy, working-day commute effects, fleet assumptions, a historical test date, and whether to buy bikes. It uses transparent intent rules and saved model outputs rather than an external generative API. It explicitly refuses to claim live station availability or exact procurement requirements.')
head('7. Business action and follow-up experiment')
p('Pilot a forecast-guided rebalancing policy in a subset of stations against matched control stations for 4–6 weeks. Record station inventory every 5–15 minutes, attempted but unsuccessful checkouts, trip start/end and duration, bike downtime, transfer costs, and service complaints. Primary metric: failed checkout attempts per 1,000 attempts; guardrails: cost per successful ride, empty-station minutes, and complaints. Randomize comparable station clusters where feasible, compare pre/post changes against controls, and avoid spillovers from bikes moved between treatment/control areas. Only consider purchasing after measuring residual stockouts despite efficient rebalancing and comparing marginal purchase costs against lost demand.')
head('8. Limitations and reproducibility')
p('This public historical dataset aggregates rental counts at citywide hourly/daily levels. It cannot identify which stations are empty, which bikes are idle, trip durations, unmet requests, current inventory, actual utilization, or purchase costs. Missing time slots may reflect absent records, not zero demand. Model evaluation uses historical weather values; production would require forecasted weather and monitoring for drift. See analysis.py, app.py, outputs/results.json, and README.md for runnable methods and outputs.')
p('Source: Fanaee-T, H. and Gama, J. (2013), “Event labeling combining ensemble detectors and background knowledge,” Progress in Artificial Intelligence. DOI: 10.1007/s13748-013-0040-3.')
doc.build(story)
print('Created report')
