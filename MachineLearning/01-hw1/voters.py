import pandas as pd 
import numpy as np

def main():
    args = getArgs()

    # it is often preferable to read directly from a zip file (I think) 
    csvFile = 'coloradovoters.{}.csv.zip'.format(args.sample)
    
    ###DF is full database
    df = pd.read_csv(csvFile, compression='infer', 
        encoding = "ISO-8859-1", low_memory=False)
    
    ###This shows the genders as male female or unknown

    #Test
    # converting the gender field to numbers... a simple 
    # example of feature engineering...
    
    ###This makes the strings into numbers
    dfGender = pd.get_dummies(df['GENDER'])
    dfParty = pd.get_dummies(df['PARTY'])
    df = pd.concat([df, dfGender], axis=1, sort=False)
    dfCorr = pd.concat([dfGender,dfParty], axis = 1, sort = False)
    
    
    writeHeatmap(dfCorr)
    
    
    #Searches for the index of the female dems another argument would show a specified column or set of columns
    fDem = df.loc[df[(df['GENDER'] == 'Female') & (df['PARTY'] == 'DEM')].index]
    fRep = df.loc[df[(df['GENDER'] == 'Female') & (df['PARTY'] == 'REP')].index]
    mDem = df.loc[df[(df['GENDER'] == 'Male') & (df['PARTY'] == 'DEM')].index]
    mRep = df.loc[df[(df['GENDER'] == 'Male') & (df['PARTY'] == 'REP')].index]
    uRep = df.loc[df[(df['GENDER'] == 'Unknown') & (df['PARTY'] == 'REP')].index]
    uDem = df.loc[df[(df['GENDER'] == 'Unknown') & (df['PARTY'] == 'DEM')].index]
    tMale= len(mRep)+len(mDem)
    tFem = len(fDem) + len(fRep)
    tUnk = len(uDem) + len(uRep)
    pMaleRep = len(mRep)/tMale
    pMaleDem = len(mDem)/tMale
    pFemRep = len(fRep)/tFem
    pFemDem = len(fDem)/tFem
    pUnkRep = len(uRep)/tUnk
    pUnkDem = len(uDem)/tUnk
    
    
    # this function uses a non-standard python library which I manually
    # installed from: https://pypi.org/project/ethnicolr/#description
    # Let me know if you can't figure out how to install from a
    # zip file.  You can comment out this line until you get it installed
    # This is just an example file anyway right?
    # 
    #correlation of party and sex
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
    #cubehelix is original
    colormap = sns.color_palette("Blues", steps)
    ax = sns.heatmap(corr, cmap=colormap, annot = True)
    
    

    plt.savefig(pngFilename)
    print(df)
    
    ##This function just prints out stuff i want to see so I know what things i put in the code

if __name__ == '__main__':
    main()
