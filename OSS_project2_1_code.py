import pandas as pd

data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

# Project #2-1 1)
print("----------Project #2-1 (1)----------")
for year in range(2015, 2019):
	year_df = data_df[data_df['year'] == year]
	for index in ['H', 'avg', 'HR', 'OBP']:
		print("%d년의 %s 상위 10명의 선수 목록" %(year, index))
		print(year_df.sort_values(by=index, ascending=False).iloc[:10, 0])

# Project #2-1 2)
print("----------Project #2-1 (2)----------")
year_2018 = data_df[data_df['year'] == 2018]
position = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']
for pos in position:
	print("2018년에서 승리 기여도가 가장 높은 %s" %pos)
	print(year_2018[year_2018['cp'] == pos].sort_values(by='war', ascending=False).iloc[0, 0])

# Project #2-1 3)
print("----------Project #2-1 (3)----------")
salary_corr = pd.Series(index=['H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG'])
for index in ['H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG']:
    salary_corr[index] = data_df[index].corr(data_df['salary'])

print(salary_corr.idxmax())