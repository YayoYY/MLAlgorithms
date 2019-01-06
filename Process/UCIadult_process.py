#!/usr/bin/env
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def process(path, skip_rows):

	raw_data = pd.read_csv(path, header=None, skiprows=skip_rows)
	raw_data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 
	                    'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
	                    'hours_per_week', 'native_country', 'salary']

	# 重复值
	df = raw_data.drop_duplicates()

	# age
	df['age'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())

	# workcalss：缺失值、dummies
	df.workclass = df.workclass.str.strip()
	df.workclass = df.workclass.replace('?', np.nan)

	# fnlwgt
	df.fnlwgt = (df.fnlwgt - df.fnlwgt.min()) / (df.fnlwgt.max() - df.fnlwgt.min())

	# education
	df.education = df.education.str.strip()
	dummies_education = pd.get_dummies(df.education, prefix='education')
	del df['education']
	df = pd.concat([df, dummies_education], axis=1)

	# education_num
	df.education_num = (df.education_num - df.education_num.min()) / (df.education_num.max() - df.education_num.min())

	# marital_status
	df.marital_status = df.marital_status.str.strip()
	dummies_marital_status = pd.get_dummies(df.marital_status, prefix='marital_status')
	del df['marital_status']
	df = pd.concat([df, dummies_marital_status], axis=1)

	# occupation：缺失值、dummies
	df.occupation = df.occupation.str.strip()
	df.occupation = df.occupation.replace('?', np.nan)

	# relationship
	df.relationship = df.relationship.str.strip()
	dummies_relationship = pd.get_dummies(df.relationship, prefix='relationship')
	del df['relationship']
	df = pd.concat([df, dummies_relationship], axis=1)

	# race
	df.race = df.race.str.strip()
	dummies_race = pd.get_dummies(df.race, prefix='race')
	del df['race']
	df = pd.concat([df, dummies_race], axis=1)

	# sex
	df.sex = df.sex.str.strip()
	dummies_sex = pd.get_dummies(df.sex, prefix='sex')
	del df['sex']
	df = pd.concat([df, dummies_sex], axis=1)

	# capital_gain
	df.capital_gain = (df.capital_gain - df.capital_gain.min()) / (df.capital_gain.max() - df.capital_gain.min())

	# capital_loss
	df.capital_loss = (df.capital_loss - df.capital_loss.min()) / (df.capital_loss.max() - df.capital_loss.min())

	# hours_per_week
	df.hours_per_week = (df.hours_per_week - df.hours_per_week.min()) / (df.hours_per_week.max() - df.hours_per_week.min())

	# native_country：缺失值、dummies
	df.native_country = df.native_country.str.strip()
	df.native_country = df.native_country.replace('?', np.nan)

	# salary
	df.salary = df.salary.str.strip()
	df.salary = df.salary.str.strip('.')
	d = {'<=50K': 0, '>50K': 1}
	df.salary = df.salary.replace(d)

	# 缺失值
	df.dropna(inplace=True)

	# workclass/occupation/native_country
	dummies_workclass = pd.get_dummies(df.workclass, prefix='workclass')
	del df['workclass']
	df = pd.concat([df, dummies_workclass], axis=1)
	dummies_occupation = pd.get_dummies(df.occupation, prefix='occupation')
	del df['occupation']
	df = pd.concat([df, dummies_occupation], axis=1)
	dummies_native_country = pd.get_dummies(df.native_country, prefix='native_country')
	del df['native_country']
	df = pd.concat([df, dummies_native_country], axis=1)

	if skip_rows:
		df.to_csv('train.csv')
	else:
		df.to_csv('test.csv')

	return df