import pandas as pd 
import numpy as np
#
#Homework is at the end of the main method
#
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
    
    
#    writeHeatmap(dfCorr)
    
    
    #Searches for the index of the female dems another argument would show a specified column or set of columns
    fDem = df.loc[df[(df['GENDER'] == 'Female') & (df['PARTY'] == 'DEM')].index]
    fRep = df.loc[df[(df['GENDER'] == 'Female') & (df['PARTY'] == 'REP')].index]
    mDem = df.loc[df[(df['GENDER'] == 'Male') & (df['PARTY'] == 'DEM')].index]
    mRep = df.loc[df[(df['GENDER'] == 'Male') & (df['PARTY'] == 'REP')].index]
    uRep = df.loc[df[(df['GENDER'] == 'Unknown') & (df['PARTY'] == 'REP')].index]
    uDem = df.loc[df[(df['GENDER'] == 'Unknown') & (df['PARTY'] == 'DEM')].index]

    
    
    # this function uses a non-standard python library which I manually
    # installed from: https://pypi.org/project/ethnicolr/#description
    # Let me know if you can't figure out how to install from a
    # zip file.  You can comment out this line until you get it installed
    # This is just an example file anyway right?
    # 
    #correlation of party and sex

    df = addEthnicityFields(df,'LAST_NAME')
#    writeHeatmap(df)
    
    
    ##################################################################
    #HOMEWORK
    ##################################################################
    #1.
    ##This adds a new column
    print("\n#################################################################\nProblem 1. \n")
    addColumn(df,"NewColumn", 0)
    print(df.columns[-1])
    
    #changes the value to 1
    df.NewColumn = 1
    print(df['NewColumn'][0])
    #renames column
    df=df.rename(columns={'NewColumn' : 'NewerColumn'})
    print(df.columns[-1])
    
    ##This Drops a column
    df = df.drop(columns = "NewerColumn")
    print(df.columns[-1])
    #################################################################
    #2.
    print("\nProblem 2. \n")
    #random sample size 3
    sample = df.sample(n=3)
    print(sample)
    
    #################################################################
    #3.
    print("\nProblem 3. \n")
    #to csv
    sample.to_csv(path_or_buf="test.csv")
    #delete
    sample = None
    print(sample)
    #read in csv
    sample = pd.read_csv(filepath_or_buffer="test.csv")
    print(sample)
    #read from zip
    csvZip = pd.read_csv("coloradovoters.1000.csv.zip", compression='infer', encoding = "ISO-8859-1", low_memory=False)
    print(csvZip.sample(n=5))
    #################################################################
    #4.
    print("\nProblem 4. \n")
    arr = [1,2,4,8]
    print(type(arr))
    #to nparray
    narr = np.asarray(arr)
    print(narr)
    print(type(narr))
    #nparray to list
    anarr = np.ndarray.tolist(narr)
    print(type(anarr))
    #################################################################
    #5.
    print("\nProblem 5. \n")
    sample["Cat"] = sample["VOTER_ID"].astype('category')
    print(sample["Cat"])
    #################################################################
    #6.
    print("\nProblem 6. \n")
    #one-hot
    sample=pd.get_dummies(sample)
    print(sample)
    #################################################################
    #7.
    print("\nProblem 7. \n")
    #these are from above and how I get the heatmap and stats
#    dfCorr = pd.concat([dfGender,dfParty], axis = 1, sort = False)
#    fDem = df.loc[df[(df['GENDER'] == 'Female') & (df['PARTY'] == 'DEM')].index]
#    fRep = df.loc[df[(df['GENDER'] == 'Female') & (df['PARTY'] == 'REP')].index]
#    mDem = df.loc[df[(df['GENDER'] == 'Male') & (df['PARTY'] == 'DEM')].index]
#    mRep = df.loc[df[(df['GENDER'] == 'Male') & (df['PARTY'] == 'REP')].index]
#    uRep = df.loc[df[(df['GENDER'] == 'Unknown') & (df['PARTY'] == 'REP')].index]
#    uDem = df.loc[df[(df['GENDER'] == 'Unknown') & (df['PARTY'] == 'DEM')].index]
    writeHeatmap(dfCorr)
    #################################################################
    #8
    #I eliminated other parties for the stats and just look at republican and dem with male female and unknown
    tMale= len(mRep)+len(mDem)
    tFem = len(fDem) + len(fRep)
    tUnk = len(uDem) + len(uRep)
    pMaleRep = len(mRep)/tMale
    pMaleDem = len(mDem)/tMale
    pFemRep = len(fRep)/tFem
    pFemDem = len(fDem)/tFem
    pUnkRep = len(uRep)/tUnk
    pUnkDem = len(uDem)/tUnk
    print("Total Males: " +str(tMale) + "\nTotal Females: " +str(tFem) + "\nTotal Unknowns: " + str(tUnk))
    print("Percent of Males who are Republican: " + str(pMaleRep*100))
    print("Percent of Females who are Republican: " + str(pFemRep*100))
    print("Percent of Unknown who are Republican: " + str(pUnkRep*100))
    print("Percent of Males who are Democrats: " + str(pMaleDem*100))
    print("Percent of Females who are Democrats: " + str(pFemDem*100))
    print("Percent of Unknown who are Democrats: " + str(pUnkDem*100))
    
    #The correlation isnt high enough to make conclusions
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

def addColumn(df, colName, col):
    df[colName] = col
if __name__ == '__main__':
    main()
