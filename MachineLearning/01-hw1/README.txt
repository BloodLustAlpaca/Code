The directory should look something like this:

-rw-r--r-- 1 tconley staff    105896 Jan 15 09:50 coloradovoters.1000.csv.zip
-rw-r--r-- 1 tconley staff   1044083 Jan 15 09:56 coloradovoters.10000.csv.zip
-rw-r--r-- 1 tconley staff 340311508 Jan 15 09:58 coloradovoters.all.csv.zip
-rwxr-xr-x 1 tconley staff       499 Jan 23 11:02 downloadVoters.sh
-rw-r--r-- 1 tconley staff     66007 Jan 23 11:14 heatmap.png
-rw-r--r-- 1 tconley staff      2374 Jan 23 11:15 voters.py


The zip files are csv (comma spearated value?) files that you can read directly into python as shown in the example file "voters.py". (you don't have to unzip them for my example) The script file shows how I downloaded the dataset in the first place.  

You may not want to download the entire dataset (coloradovoters.all.csv.zip), at first, since it is really large.  There are some smaller sample files that you can use. The file (coloradovoters.1000.csv.zip) is a 1000 random sample of the dataset, there is also a file with 10000 samples.  You can use these files instead of the whole file for speed and testing.

See if you can figure out what voters.py is doing and see if you can do more with the data.