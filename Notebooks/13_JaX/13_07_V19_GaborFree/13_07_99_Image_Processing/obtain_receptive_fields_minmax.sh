# python -u 13_00_99b_Receptive_Field_Loop.py -p Receptive_Fields_MinMax/GDN_0 -l GDNGamma_0 --min-max
# python -u 13_00_99b_Receptive_Field_Loop.py -p Receptive_Fields_MinMax/Color -l Color --min-max
# python -u 13_00_99b_Receptive_Field_Loop.py -p Receptive_Fields_MinMax/GDN_1 -l GDN_0 --min-max
# python -u 13_00_99b_Receptive_Field_Loop.py -p Receptive_Fields_MinMax/CenterSurroundLogSigmaK_0 -l CenterSurroundLogSigmaK_0 --min-max
# python -u 13_00_99b_Receptive_Field_Loop.py -p Receptive_Fields_MinMax/GDN_2 -l GDNGaussian_0 --min-max
python -u 13_00_99b_Receptive_Field_Loop.py -p Receptive_Fields_MinMax/GaborLayerGammaRepeat_0 -l GaborLayerGammaRepeat_0 --min-max -b 8
python -u 13_00_99b_Receptive_Field_Loop.py -p Receptive_Fields_MinMax/GDN_Final -l GDNSpatioFreqOrient_0 --min-max -b 12
