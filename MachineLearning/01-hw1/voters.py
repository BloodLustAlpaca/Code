import pandas as pd 
import numpy as np

def main():
	args = getArgs()

	# it is often preferable to read directly from a zip file (I think) 
	csvFile = 'coloradovoters.{}.csv.zip'.format(args.sample)
	df = pd.read_csv(csvFile, compression='infer', 
		encoding = "ISO-8859-1", low_memory=False)

	# converting the gender field to numbers... a simple 
	# example of feature engineering...
	dfGender = pd.get_dummies(df['GENDER'])
	df = pd.concat([df, dfGender], axis=1, sort=False)

	print(df.columns)

	# this function uses a non-standard python library which I manually
	# installed from: https://pypi.org/project/ethnicolr/#description
	# Let me know if you can't figure out how to install from a
	# zip file.  You can comment out this line until you get it installed
	# This is just an example file anyway right?
	# 
	df = addEthnicityFields(df,'LAST_NAME')

	writeHeatmap(df)


def getArgs():
	import argparse
	parser = argparse.ArgumentParser(description = 'CS whatever -- Homework 1 -- Thomas Conley',add_help=True)

	parser.add_argument('--verbose', 
		help='Verbose mode.', 
		action='store_true')

	parser.add_argument('--sample', 
		help='Sample the dataset.', 
		type=str,
		choices=['1000','10000','all'],
		default='10000'
		)

	args = parser.parse_args()

	return args

# add add ethnicity fields to a df based on a name field
def addEthnicityFields(df,namefield):
	# https://pypi.org/project/ethnicolr/#description
	import ethnicolr
	# use only the last word of the field for analysis
	df['ethname'] = df[namefield].transform(lambda t: t.split()[-1])
	# convert using library function
	df = ethnicolr.census_ln(df,'ethname')
	# drop the temporary column
	df = df.drop(columns=['ethname'])

	newfields = ['pctwhite', 'pctblack', 'pctapi', 'pctaian', 'pct2prace', 'pcthispanic']
	for fieldname in newfields:
		df[fieldname] = pd.to_numeric(df[fieldname].astype(str), errors='coerce').astype(float)

	return df

def writeHeatmap(df, pngFilename='heatmap.png'):
	import matplotlib.pyplot as plt
	import seaborn as sns; 
	sns.set()

	# its a heat map of correlation  
	corr = df.corr()

	# Set up the matplotlib figure
	f, ax = plt.subplots(figsize=(11, 9))

	# Generate a custom diverging color map
	steps = 50
	colormap = sns.color_palette("cubehelix", steps)
	ax = sns.heatmap(corr, cmap=colormap)

	plt.savefig(pngFilename)

if __name__ == '__main__':
	main()
