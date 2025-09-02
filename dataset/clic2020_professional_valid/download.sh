wget https://data.vision.ee.ethz.ch/cvl/clic/professional_valid_2020.zip
unzip professional_valid_2020.zip -d ./
mv valid/* ./
rm -rf valid
rm professional_valid_2020.zip
rm -rf __MACOSX