# Before using this script, make sure noise100.txt file is in this directory.
# This script will generate a noise100.SEQ in /input/seqfiles which can be read by the sPCA script.

java -classpath target/sparkPCA-1.0.jar -DInput=noise100.txt -DInputFmt=DENSE -DOutput=input/seqfiles -DCardinality=4662 org.qcri.sparkpca.FileFormat
mv input/seqfiles/noise100.txt.seq input/seqfiles/noise100.seq