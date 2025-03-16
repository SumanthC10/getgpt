grep "ENG" MRCONSO.RRF | grep "MSH" > MRCONSO_ENG_MSH.RRF


awk -F"|" '{
  for(i=1; i<=NF; i++){
    if($i=="MSH"){
      print $1 "|" $(i-1);
      break;
    }
  }
}' MRCONSO_ENG_MSH.RRF > extracted.txt
(echo "umls_code,mesh_code" && sort extracted.txt | uniq | sed 's/|/,/g') > mesh_umls.csv
