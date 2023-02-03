#!/bin/bash

# Script to download, extract and arrange SEN12MS-CR-TS and SEN12MS-CR.
# Make this script executable (by running: chmod +x dl_data.sh), 
# then give it a run (by calling: ./dl_data.sh) and
# follow the prompts in order to get the desired data.    
    
clear
echo "This script is for downloading the SEN12MS-CR-TS data set for cloud removal in satellite data."
echo See the associated paper: Ebel et al \(2022\) \'SEN12MS-CR-TS: A Remote Sensing Data Set for Multi-modal Multi-temporal Cloud Removal\'
echo -e 'Click \e]8;;https://patricktum.github.io/cloud_removal/\ahere\e]8;;\a for more information'
echo
echo

while true; do
    read -p "Do you wish to download the multitemporal SEN12MS-CR-TS data set? " yn
    case $yn in
        [Yy]* ) SEN12MSCRTS=true; break;;
        [Nn]* ) SEN12MSCRTS=false; break;;
        * ) echo "Please answer yes or no.";;
    esac
done

if [ "$SEN12MSCRTS" = "true" ]; then
	while true; do
		read -p "What regions would you like to download? [all|africa|america|asiaEast|asiaWest|europa] " region
		case $region in
			all|africa|america|asiaEast|asiaWest|europa ) reg=$region; break;;
		    * ) echo "Please answer [all|africa|america|asiaEast|asiaWest|europa].";;
		esac
	done
fi

while true; do
    read -p "Do you wish to also download the monotemporal SEN12MS-CR data set (all regions)? " yn
    case $yn in
        [Yy]* ) SEN12MSCR=true; break;;
        [Nn]* ) SEN12MSCR=false; break;;
        * ) echo "Please answer yes or no.";;
    esac
done

while true; do
	read -p "Do you wish to also download the Sentinel-1 radar data associated with your previous choices? " yn
	case $yn in
	    [Yy]* ) S1=true; break;;
	    [Nn]* ) S1=false; break;;
	    * ) echo "Please answer yes or no.";;
	esac
done

declare -A url_dict # holding links to data
declare -A vol_dict # bookkeeping size of data

echo "Please enter the path to download and extract the data to: "
read dl_extract_to


echo
echo 
if [ "$SEN12MSCRTS" = "true" ]; then
	
	echo "Downloading SEN12MS-CR-TS data set."
	mkdir -p $dl_extract_to'/SEN12MSCRTS'
	
	# train split
	case $region in
		'all') 		url_dict['multi_s2_africa']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s2_africa.tar.gz'
					vol_dict['multi_s2_africa']='98233900'

					url_dict['multi_s2_america']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s2_america.tar.gz'
					vol_dict['multi_s2_america']='110245004'

					url_dict['multi_s2_asiaEast']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s2_asiaEast.tar.gz'
					vol_dict['multi_s2_asiaEast']='113948560'

					url_dict['multi_s2_asiaWest']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s2_asiaWest.tar.gz'
					vol_dict['multi_s2_asiaWest']='96082796'

					url_dict['multi_s2_europa']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s2_europa.tar.gz'
					vol_dict['multi_s2_europa']='196669740'
					;;
		'africa') 	url_dict['multi_s2_africa']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s2_africa.tar.gz'
					vol_dict['multi_s2_africa']='98233900'
					;;
		'america') 	url_dict['multi_s2_america']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s2_america.tar.gz'
					vol_dict['multi_s2_america']='110245004'
					;;
		'asiaEast') url_dict['multi_s2_asiaEast']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s2_asiaEast.tar.gz'
					vol_dict['multi_s2_asiaEast']='113948560'
					;;
		'asiaWest') url_dict['multi_s2_asiaWest']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s2_asiaWest.tar.gz'
					vol_dict['multi_s2_asiaWest']='96082796'
					;;
		'europa') 	url_dict['multi_s2_europa']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s2_europa.tar.gz'
					vol_dict['multi_s2_europa']='196669740'
					;;
	esac


	# test split
	case $region in
		'all') 		url_dict['multi_s2_africa_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s2_africa_test.tar.gz'
					vol_dict['multi_s2_africa_test']='25421744'

					url_dict['multi_s2_america_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s2_america_test.tar.gz'
					vol_dict['multi_s2_america_test']='25421824'

					url_dict['multi_s2_asiaEast_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s2_asiaEast_test.tar.gz'
					vol_dict['multi_s2_asiaEast_test']='40534760'

					url_dict['multi_s2_asiaWest_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s2_asiaWest_test.tar.gz'
					vol_dict['multi_s2_asiaWest_test']='15012924'

					url_dict['multi_s2_europa_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s2_europa_test.tar.gz'
					vol_dict['multi_s2_europa_test']='79568460'
					;;
		'africa') 	url_dict['multi_s2_africa_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s2_africa_test.tar.gz'
					vol_dict['multi_s2_africa_test']='25421744'
					;;
		'america') 	url_dict['multi_s2_america_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s2_america_test.tar.gz'
					vol_dict['multi_s2_america_test']='25421824'
					;;
		'asiaEast') url_dict['multi_s2_asiaEast_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s2_asiaEast_test.tar.gz'
					vol_dict['multi_s2_asiaEast_test']='40534760'
					;;
		'asiaWest') url_dict['multi_s2_asiaWest_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s2_asiaWest_test.tar.gz'
					vol_dict['multi_s2_asiaWest_test']='15012924'
					;;
		'europa') 	url_dict['multi_s2_europa_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s2_europa_test.tar.gz'
					vol_dict['multi_s2_europa_test']='79568460'
					;;
	esac


	if [ "$S1" = "true" ]; then
		# train split
		case $region in
		'all') 		url_dict['multi_s1_africa']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s1_africa.tar.gz'
					vol_dict['multi_s1_africa']='60544524'

					url_dict['multi_s1_america']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s1_america.tar.gz'
					vol_dict['multi_s1_america']='67947416'

					url_dict['multi_s1_asiaEast']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s1_asiaEast.tar.gz'
					vol_dict['multi_s1_asiaEast']='70230104'

					url_dict['multi_s1_asiaWest']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s1_asiaWest.tar.gz'
					vol_dict['multi_s1_asiaWest']='59218848'

					url_dict['multi_s1_europa']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s1_europa.tar.gz'
					vol_dict['multi_s1_europa']='121213836'
					;;
		'africa') 	url_dict['multi_s1_africa']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s1_africa.tar.gz'
					vol_dict['multi_s1_africa']='60544524'
					;;
		'america') 	url_dict['multi_s1_america']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s1_america.tar.gz'
					vol_dict['multi_s1_america']='67947416'
					;;
		'asiaEast') url_dict['multi_s1_asiaEast']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s1_asiaEast.tar.gz'
					vol_dict['multi_s1_asiaEast']='70230104'
					;;
		'asiaWest') url_dict['multi_s1_asiaWest']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s1_asiaWest.tar.gz'
					vol_dict['multi_s1_asiaWest']='59218848'
					;;
		'europa') 	url_dict['multi_s1_europa']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s1_europa.tar.gz'
					vol_dict['multi_s1_europa']='121213836'
					;;
		esac
		
		
		# test split
		case $region in
			'all') 		url_dict['multi_s1_africa_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s1_africa_test.tar.gz'
						vol_dict['multi_s1_africa_test']='15668120'

						url_dict['multi_s1_america_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s1_america_test.tar.gz'
						vol_dict['multi_s1_america_test']='15668160'

						url_dict['multi_s1_asiaEast_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s1_asiaEast_test.tar.gz'
						vol_dict['multi_s1_asiaEast_test']='24982736'

						url_dict['multi_s1_asiaWest_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s1_asiaWest_test.tar.gz'
						vol_dict['multi_s1_asiaWest_test']='9252904'

						url_dict['multi_s1_europa_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s1_europa_test.tar.gz'
						vol_dict['multi_s1_europa_test']='49040432'
						;;
			'africa') 	url_dict['multi_s1_africa_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s1_africa_test.tar.gz'
						vol_dict['multi_s1_africa_test']='15668120'
						;;
			'america') 	url_dict['multi_s1_america_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s1_america_test.tar.gz'
						vol_dict['multi_s1_america_test']='15668160'
						;;
			'asiaEast') url_dict['multi_s1_asiaEast_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s1_asiaEast_test.tar.gz'
						vol_dict['multi_s1_asiaEast_test']='24982736'
						;;
			'asiaWest') url_dict['multi_s1_asiaWest_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s1_asiaWest_test.tar.gz'
						vol_dict['multi_s1_asiaWest_test']='9252904'
						;;
			'europa') 	url_dict['multi_s1_europa_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s1_europa_test.tar.gz'
						vol_dict['multi_s1_europa_test']='49040432'
						;;
		esac
	fi
fi


# mono-temporal data (all regions)
if [ "$SEN12MSCR" = "true" ]; then
	echo "Also downloading SEN12MS-CR data set."
	mkdir -p $dl_extract_to'/SEN12MSCR'
	url_dict['mono_s2_spring']='https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1158_spring_s2.tar.gz'
	vol_dict['mono_s2_spring']='48568904'
	
	url_dict['mono_s2_summer']='https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1868_summer_s2.tar.gz'
	vol_dict['mono_s2_summer']='56425520'
	
	url_dict['mono_s2_fall']='https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1970_fall_s2.tar.gz'
	vol_dict['mono_s2_fall']='68291864'
	
	url_dict['mono_s2_winter']='https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs2017_winter_s2.tar.gz'
	vol_dict['mono_s2_winter']='30580552'
	
	url_dict['mono_s2_cloudy_spring']='https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1158_spring_s2_cloudy.tar.gz'
	vol_dict['mono_s2_cloudy_spring']='48569368'
	
	url_dict['mono_s2_cloudy_summer']='https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1868_summer_s2_cloudy.tar.gz'
	vol_dict['mono_s2_cloudy_summer']='56426004'
	
	url_dict['mono_s2_cloudy_fall']='https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1970_fall_s2_cloudy.tar.gz'
	vol_dict['mono_s2_cloudy_fall']='68292448'
	
	url_dict['mono_s2_cloudy_winter']='https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs2017_winter_s2_cloudy.tar.gz'
	vol_dict['mono_s2_cloudy_winter']='30580812'
	
	# S1 data of SEN12MS-CR
	if [ "$S1" = "true" ]; then
		echo "Also downloading associated S1 data."
		url_dict['mono_s1_spring']='https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1158_spring_s1.tar.gz'
		vol_dict['mono_s1_spring']='15026120'
		
		url_dict['mono_s1_summer']='https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1868_summer_s1.tar.gz'
		vol_dict['mono_s1_summer']='17456784'
		
		url_dict['mono_s1_fall']='https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1970_fall_s1.tar.gz'
		vol_dict['mono_s1_fall']='21127832'
		
		url_dict['mono_s1_winter']='https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs2017_winter_s1.tar.gz'
		vol_dict['mono_s1_winter']='9460956'
	fi
fi

req=0
# integrate file size across archives
for key in "${!vol_dict[@]}"; do
	# for each archive: sum up
	curr=${vol_dict[$key]}
	req=$((req+curr))
done

echo
echo
# df -h $dl_extract_to
avail=$(df $dl_extract_to | awk 'NR==2 { print $4 }')
if (( avail < req )); then
	echo "Not enough space (512-byte disk sectors) on path "$dl_extract_to". Available "$avail". Required "$req #>&2
	exit 1
else
	echo "Consuming "$req" of "$avail" (512-byte disk sectors) on path "$dl_extract_to
fi
echo
echo

# download each archive individually, then extract individually

# fetch the actual data
for key in "${!url_dict[@]}"; do
    url=${url_dict[$key]}
    filename=$(basename "$url")
    filename=${filename:7}
    # download
    wget --no-check-certificate -c -O $dl_extract_to'/'$filename ${url_dict[$key]}
    # unzip and delete archive
    tar --extract --file $dl_extract_to'/'$filename -C $dl_extract_to
    rm $dl_extract_to'/'$filename
done

# move the extracted data to its respective place (this may take a while, because we use rsync rather than mv)
echo "Moving data in place, please don't stop this process."
for key in "${!url_dict[@]}"; do
    url=${url_dict[$key]}
    filename=$(basename "$url")
    filename=${filename:7:-7} # remove base URL and trailing *.tar.gz
    if [[ ${url_dict[$key]} == *"m1554803"* ]]; then
    	# move to SEN12MSCR directory
	  	mv $dl_extract_to'/'$filename $dl_extract_to'/SEN12MSCR'
  	elif [[ ${url_dict[$key]} == *"m1639953"* ]]; then
		# move train ROI to SEN12MSCRTS directory
		no_prefix_filename=${filename:3}
		rsync -a -remove-source-files $dl_extract_to'/'$no_prefix_filename/* $dl_extract_to'/SEN12MSCRTS' 2>/dev/null
		rm -rf $dl_extract_to'/'$no_prefix_filename
	else
		# move test ROI to SEN12MSCRTS directory
		rsync -a -remove-source-files $dl_extract_to'/'$filename/* $dl_extract_to'/SEN12MSCRTS'
		rm -rf $dl_extract_to'/'$filename
	fi
done

echo
echo "Completed downloading, extracting and moving data! Enjoy :)"
