#!/bin/bash

# download the zip files, I did this in stages
for i in {1..8}
do
	wget "http://coloradovoters.info/downloads/20170801/Registered_Voters_List_ Part$i.zip"
	open "Registered_Voters_List_ Part$i.zip"
	mv "Registered_Voters_List_ Part$i.txt" "Part$i.txt"
done

# concat into a single csv
head -1 Part1.txt > coloradovoters.csv
for i in {1..8}
do
	# append the file without header line
	cat Part$i.txt | grep -v '^"VOTER_ID"' >> coloradovoters.csv
done

# remove the part files
rm Part?.txt