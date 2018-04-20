if [ ! -d "./outputs" ]; then
  mkdir "./outputs"
fi
find "./" -type f -name "*.xlsx" | while read line; do
  foldername=$(basename $(dirname $line))
  filename=$(basename "$line" .xlsx);
  if [ ! -d "./outputs/$foldername" ]; then
     mkdir "./outputs/$foldername"
  fi
  in2csv $line > "./outputs/$foldername/$filename"
done
