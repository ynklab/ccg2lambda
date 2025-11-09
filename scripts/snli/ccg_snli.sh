cd /home/$USER/ccg2lambda/snli
echo "Array Job Index: ${PMI_RANK}/${PMI_SIZE}"
python ccg_txt_snli.py ${PMI_RANK} 1>log/${PMI_RANK}.out 2>log/${PMI_RANK}.err
