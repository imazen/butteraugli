BEGIN{FS="\t"}
NR>1 && !($2==7&&$3==9){k=$5"\t"$6; n[k]++; if($13==0)ex[k]++; if($9>md[k])md[k]=$9; r=$9/$7; if(r>mr[k])mr[k]=r; if($12>mp[k])mp[k]=$12; if($14>ma[k])ma[k]=$14; mm[k]+=$13; px[k]+=$2*$3}
END{print "arm\tnumerics\tpairs\tmaps_bitexact\tmax|dscore|\tmax_rel_dscore\tmax|dpnorm3|\tmap_max_abs\tmap_px_differing"; for(k in n) printf "%s\t%d\t%d\t%.3e\t%.3e\t%.3e\t%.3e\t%.2f%%\n",k,n[k],ex[k]+0,md[k],mr[k],mp[k],ma[k],100*mm[k]/px[k]}
