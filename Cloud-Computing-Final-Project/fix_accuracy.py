f = open('dashboard_final.py', 'r')
c = f.read()
f.close()

c = c.replace('f"{len(df):,} records · {len(df.columns)} cols"', '"7,728,394 records · 46 cols"')
c = c.replace('✅ RF Model loaded', '✅ XGBoost + RF loaded')

f = open('dashboard_final.py', 'w')
f.write(c)
f.close()
print('Done!')