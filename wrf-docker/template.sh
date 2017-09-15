#!/bin/bash
#####################################################################
#                      ts & Skarozis & BP 06/03/2013                     #
#####################################################################
chmod 666 cluster*
date=!@#date
#echo "Give hour of wrfout file in hh"
h=!@#h
#echo "Give simulations hours of wrfout file"
sim=11
#-------------------------------------------------------------------------------
NC=which ncks
if [ -x $NC ]; then
    NCO=ncks
    echo "ncks exists, Linux version"
elif [ -x ncks ]; then
    NCO=ncks
    echo "ncks exists, Linux version"
elif [ -x ncks.exe ]; then
    NCO=ncks.exe
    echo "ncks.exe exists, Cygwin version"
else
    echo "ncks does not exist... Exit"
    exit
fi
#-------------------------------------------------------------------------------
filename=!@#file
#-------------------------------------------------------------------------------
#mkdir splits/
let "j=$sim"
for i in $(eval echo {0.."$j"})
do
  let "hour=$h+6*$i"
  wrf=`date "-d $date $hour hours" +%Y-%m-%d_%H`":00:00.nc"
  echo $wrf
  $NCO -d Time,$i $filename met_em.d01.$wrf 
done
#-------------------------------------------------------------------------------
echo "Split netcdf complete"
#-------------------------------------------------------------------------------
