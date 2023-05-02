#!/usr/bin/env python
# coding: utf-8

# # Data for labs

# In[1]:


import numpy as np

# Normal coordinate to the flame front, $x\,[m]$.

# In[2]:


x = np.array([0.        , 0.004     , 0.008     , 0.01      , 0.011     ,
       0.012     , 0.0125    , 0.01275   , 0.013     , 0.013125  ,
       0.01325   , 0.013375  , 0.0134375 , 0.0135    , 0.0135625 ,
       0.013625  , 0.01365625, 0.0136875 , 0.01371875, 0.01373438,
       0.01375   , 0.01376563, 0.01378125, 0.01379688, 0.0138125 ,
       0.01382031, 0.01382813, 0.01383594, 0.01384375, 0.01385156,
       0.01385938, 0.01386719, 0.013875  , 0.01388281, 0.01389063,
       0.01389844, 0.01390234, 0.01390625, 0.01391016, 0.01391406,
       0.01391797, 0.01392188, 0.01392578, 0.01392969, 0.01393359,
       0.0139375 , 0.01394531, 0.01395313, 0.01396094, 0.01396875,
       0.01397656, 0.01398438, 0.01399219, 0.014     , 0.01400391,
       0.01400781, 0.01401172, 0.01401563, 0.01401953, 0.01402344,
       0.01402734, 0.01403125, 0.01403516, 0.01403906, 0.01404687,
       0.01405469, 0.0140625 , 0.01407031, 0.01407812, 0.01408203,
       0.01408594, 0.01408984, 0.01409375, 0.01409766, 0.01410156,
       0.01410547, 0.01410937, 0.01411328, 0.01411719, 0.01412109,
       0.014125  , 0.01412891, 0.01413281, 0.01413672, 0.01414062,
       0.01414453, 0.01414844, 0.01415234, 0.01415625, 0.01416016,
       0.01416406, 0.01416797, 0.01417187, 0.01417578, 0.01417969,
       0.01418359, 0.0141875 , 0.01419141, 0.01419531, 0.01419922,
       0.01420312, 0.01420703, 0.01421094, 0.01421484, 0.01421875,
       0.01422266, 0.01422656, 0.01423047, 0.01423437, 0.01423828,
       0.01424219, 0.01424609, 0.01425   , 0.01425391, 0.01425781,
       0.01426172, 0.01426562, 0.01426953, 0.01427344, 0.01427734,
       0.01428125, 0.01428516, 0.01428906, 0.01429297, 0.01429687,
       0.01430078, 0.01430469, 0.01430859, 0.0143125 , 0.01431641,
       0.01432031, 0.01432422, 0.01432812, 0.01433203, 0.01433594,
       0.01433984, 0.01434375, 0.01434766, 0.01435156, 0.01435937,
       0.01436719, 0.014375  , 0.01438281, 0.01439062, 0.01439844,
       0.01440625, 0.01441406, 0.01442187, 0.0144375 , 0.01445312,
       0.01446875, 0.01448437, 0.0145    , 0.01453125, 0.0145625 ,
       0.01459375, 0.014625  , 0.0146875 , 0.01475   , 0.0148125 ,
       0.014875  , 0.0149375 , 0.015     , 0.0150625 , 0.015125  ,
       0.01525   , 0.015375  , 0.0155    , 0.015625  , 0.01575   ,
       0.015875  , 0.016     , 0.01625   , 0.0165    , 0.01675   ,
       0.017     , 0.0175    , 0.018     , 0.0185    , 0.019     ,
       0.02      , 0.021     , 0.022     , 0.023     , 0.024     ,
       0.025     , 0.026     , 0.027     , 0.028     , 0.03      ,
       0.032     , 0.034     , 0.036     , 0.04      ], dtype=np.float32)


# Local temperature, $T\,[K]$

# In[ ]:


T = np.array([ 300.        ,  300.        ,  300.00000003,  300.00000063,
        300.00000841,  300.00014726,  300.00109364,  300.00455456,
        300.02257472,  300.05998917,  300.17593106,  300.53462853,
        300.99476431,  301.9329261 ,  303.83845714,  307.68265088,
        311.04864547,  316.0649382 ,  323.47486835,  328.47153286,
        334.59627593,  342.06338193,  351.11036566,  361.99363529,
        374.98159233,  382.38092603,  390.44233418,  399.20165562,
        408.69322588,  418.94944421,  430.0003733 ,  441.87338583,
        454.59286766,  468.1799847 ,  482.65251625,  498.02475417,
        506.05702119,  514.32294239,  522.82339624,  531.55890751,
        540.5298866 ,  549.73641552,  559.17845193,  568.85565248,
        578.76754203,  588.9133726 ,  609.89377004,  631.77943955,
        654.55862611,  678.21709038,  702.73831227,  728.10368605,
        754.29269161,  781.28301936,  795.07397397,  809.05907396,
        823.23488682,  837.59784753,  852.14425316,  866.8702432 ,
        881.77179578,  896.84469629,  912.0845373 ,  927.48667464,
        958.7568708 ,  990.60703033, 1022.99154718, 1055.85850427,
       1089.14970771, 1105.93171078, 1122.79574203, 1139.73234215,
       1156.73155359, 1173.78296537, 1190.87566181, 1207.99826258,
       1225.1388926 , 1242.28521255, 1259.42441587, 1276.54324481,
       1293.62801998, 1310.66463723, 1327.63863454, 1344.53516568,
       1361.33910822, 1378.03501096, 1394.60724582, 1411.03992747,
       1427.31711034, 1443.42268056, 1459.3405977 , 1475.05476083,
       1490.54929257, 1505.80838246, 1520.8166088 , 1535.55876358,
       1550.02020517, 1564.18666977, 1578.04464566, 1591.58117596,
       1604.78424334, 1617.64256829, 1630.14599149, 1642.28527071,
       1654.05244734, 1665.44064349, 1676.44439925, 1687.05947   ,
       1697.28312222, 1707.11392975, 1716.55201806, 1725.59885768,
       1734.25745018, 1742.53211722, 1750.42862561, 1757.95397142,
       1765.11644624, 1771.92541774, 1778.39134349, 1784.52555162,
       1790.3402124 , 1795.84812458, 1801.06265741, 1805.99754949,
       1810.66683481, 1815.08466108, 1819.26521248, 1823.22255355,
       1826.97055919, 1830.52278797, 1833.89242689, 1837.0921957 ,
       1840.13431026, 1843.03041693, 1845.79157539, 1848.42822001,
       1850.95016036, 1853.36656511, 1855.68597683, 1860.05787204,
       1864.13195665, 1867.95497027, 1871.56588684, 1874.9969009 ,
       1878.27440716, 1881.41992396, 1884.45093053, 1887.38160338,
       1892.97686883, 1898.2875107 , 1903.3572074 , 1908.21631216,
       1912.88682673, 1921.71359486, 1929.96656914, 1937.71362015,
       1945.00861374, 1958.37936657, 1970.47970631, 1981.51312692,
       1991.63586543, 2000.96992193, 2009.61194883, 2017.6393798 ,
       2025.11461061, 2038.56063184, 2050.50349892, 2061.19070363,
       2070.81357718, 2079.52204697, 2087.43385577, 2094.64001709,
       2107.13554244, 2117.87248395, 2127.18672756, 2135.31587329,
       2148.46851032, 2159.13268834, 2167.93017313, 2175.27122511,
       2186.27814699, 2194.65061805, 2201.16668904, 2206.32813257,
       2210.47343948, 2213.83945067, 2216.59686711, 2218.8712552 ,
       2220.75333576, 2223.47321959, 2225.44055421, 2226.87317309,
       2227.89362275, 2227.89362275], dtype=np.float32)


# Local heat release rate, ${HRR}\,[J/s]$

# In[3]:


HRR = np.array([-7.38732051e+02,  6.92375373e-05,  7.10934971e-05,  2.95037750e-05,
        1.91952342e-05,  1.70848410e-05,  7.30556653e-05,  1.04744118e-03,
        2.49996002e-02,  2.05889802e-01,  1.84021766e+00,  1.69233787e+01,
        6.27520428e+01,  2.42617270e+02,  9.60081664e+02,  3.81607770e+03,
        8.06421578e+03,  1.73402476e+04,  3.75299111e+04,  5.61547488e+04,
        8.45043963e+04,  1.27650303e+05,  1.93372599e+05,  2.93787936e+05,
        4.48165611e+05,  5.58294979e+05,  6.97998300e+05,  8.76618778e+05,
        1.10696354e+06,  1.40677463e+06,  1.80059385e+06,  2.32200309e+06,
        3.01611853e+06,  3.94212875e+06,  5.17561715e+06,  6.81038636e+06,
        7.83670236e+06,  9.01174133e+06,  1.03541796e+07,  1.18838814e+07,
        1.36218306e+07,  1.55901701e+07,  1.78120436e+07,  2.03115751e+07,
        2.31136589e+07,  2.62439916e+07,  3.34477620e+07,  4.22310141e+07,
        5.27857771e+07,  6.52986680e+07,  7.99510712e+07,  9.69282624e+07,
        1.16439789e+08,  1.38749961e+08,  1.51308434e+08,  1.64744772e+08,
        1.79132813e+08,  1.94557520e+08,  2.11116330e+08,  2.28919779e+08,
        2.48091865e+08,  2.68770150e+08,  2.91105953e+08,  3.15265725e+08,
        3.68647730e+08,  4.31167196e+08,  5.04324687e+08,  5.89661693e+08,
        6.88673356e+08,  7.44736983e+08,  8.04852128e+08,  8.69184760e+08,
        9.37879758e+08,  1.01105814e+09,  1.08881693e+09,  1.17122367e+09,
        1.25831656e+09,  1.35009851e+09,  1.44653812e+09,  1.54756361e+09,
        1.65306451e+09,  1.76288551e+09,  1.87682869e+09,  1.99464729e+09,
        2.11604823e+09,  2.24068599e+09,  2.36816501e+09,  2.49803432e+09,
        2.62978962e+09,  2.76286931e+09,  2.89665628e+09,  3.03047644e+09,
        3.16360038e+09,  3.29524542e+09,  3.42457754e+09,  3.55071805e+09,
        3.67274614e+09,  3.78971086e+09,  3.90063474e+09,  4.00453130e+09,
        4.10041008e+09,  4.18729921e+09,  4.26425191e+09,  4.33037292e+09,
        4.38482613e+09,  4.42686290e+09,  4.45582987e+09,  4.47119645e+09,
        4.47256144e+09,  4.45967648e+09,  4.43244952e+09,  4.39096128e+09,
        4.33546358e+09,  4.26638629e+09,  4.18432914e+09,  4.09005778e+09,
        3.98448863e+09,  3.86867444e+09,  3.74378290e+09,  3.61107389e+09,
        3.47187389e+09,  3.32754848e+09,  3.17947575e+09,  3.02901810e+09,
        2.87749821e+09,  2.72617419e+09,  2.57622099e+09,  2.42871195e+09,
        2.28460757e+09,  2.14474500e+09,  2.00983453e+09,  1.88045704e+09,
        1.75706747e+09,  1.63999880e+09,  1.52947081e+09,  1.42559880e+09,
        1.32840548e+09,  1.23783257e+09,  1.15375508e+09,  1.00507042e+09,
        8.79221647e+08,  7.73751318e+08,  6.86055629e+08,  6.13562255e+08,
        5.53855185e+08,  5.04749791e+08,  4.64328389e+08,  4.30948107e+08,
        3.81390082e+08,  3.46160158e+08,  3.20067829e+08,  2.99807703e+08,
        2.83302832e+08,  2.57767924e+08,  2.37320563e+08,  2.20050502e+08,
        2.05072112e+08,  1.80496918e+08,  1.60808590e+08,  1.44709936e+08,
        1.31317578e+08,  1.20003273e+08,  1.10313581e+08,  1.01918824e+08,
        9.45927007e+07,  8.24026490e+07,  7.25707689e+07,  6.44784429e+07,
        5.77175718e+07,  5.20012228e+07,  4.71213535e+07,  4.29345595e+07,
        3.61789111e+07,  3.09137287e+07,  2.67279826e+07,  2.33560291e+07,
        1.83959521e+07,  1.48396587e+07,  1.22018046e+07,  1.01963564e+07,
        7.49191742e+06,  5.68032173e+06,  4.40946766e+06,  3.48555720e+06,
        2.79465489e+06,  2.26617529e+06,  1.85446588e+06,  1.52903346e+06,
        1.26912367e+06,  9.07390080e+05,  6.56050093e+05,  4.78667427e+05,
        3.55995233e+05,  3.55995233e+05], dtype=np.float32)


# Local mass fraction of the OH radical, $Y_{OH}\,[-]$

# In[4]:


Y_OH = np.array([-7.56688725e-14, -1.84501385e-16,  1.43428723e-16,  1.43786933e-16,
        1.45327624e-16,  1.94135320e-16,  5.56218605e-16,  3.90464131e-15,
        7.01257397e-14,  6.04405302e-13,  5.03487523e-12,  4.08874494e-11,
        1.44455421e-10,  4.85452413e-10,  1.57141146e-09,  4.81653376e-09,
        8.64443851e-09,  1.53654779e-08,  2.71034894e-08,  3.63950225e-08,
        4.89981382e-08,  6.61748047e-08,  8.97279547e-08,  1.22264948e-07,
        1.67606189e-07,  1.97580778e-07,  2.33458056e-07,  2.76494660e-07,
        3.28240187e-07,  3.90612598e-07,  4.65991878e-07,  5.57332733e-07,
        6.68293837e-07,  8.03376195e-07,  9.68057174e-07,  1.16890175e-06,
        1.28683818e-06,  1.41710925e-06,  1.56083566e-06,  1.71918633e-06,
        1.89336508e-06,  2.08459591e-06,  2.29410455e-06,  2.52309900e-06,
        2.77274655e-06,  3.04415057e-06,  3.64941937e-06,  4.35158861e-06,
        5.15730631e-06,  6.07193076e-06,  7.10025427e-06,  8.24787064e-06,
        9.52326826e-06,  1.09406502e-05,  1.17186065e-05,  1.25425615e-05,
        1.34171590e-05,  1.43481060e-05,  1.53422721e-05,  1.64077860e-05,
        1.75541340e-05,  1.87922499e-05,  2.01346086e-05,  2.15953084e-05,
        2.48891817e-05,  2.88544518e-05,  3.36658333e-05,  3.95303266e-05,
        4.66900805e-05,  5.08959345e-05,  5.55433191e-05,  6.06759372e-05,
        6.63405922e-05,  7.25873419e-05,  7.94696082e-05,  8.70443211e-05,
        9.53720065e-05,  1.04516906e-04,  1.14547027e-04,  1.25534220e-04,
        1.37554175e-04,  1.50686435e-04,  1.65014312e-04,  1.80624806e-04,
        1.97608423e-04,  2.16058958e-04,  2.36073173e-04,  2.57750417e-04,
        2.81192124e-04,  3.06501221e-04,  3.33781417e-04,  3.63136366e-04,
        3.94668725e-04,  4.28479048e-04,  4.64664607e-04,  5.03318039e-04,
        5.44525945e-04,  5.88367336e-04,  6.34912070e-04,  6.84219185e-04,
        7.36335266e-04,  7.91292781e-04,  8.49108520e-04,  9.09782076e-04,
        9.73294512e-04,  1.03960717e-03,  1.10866074e-03,  1.18037454e-03,
        1.25464617e-03,  1.33135145e-03,  1.41034476e-03,  1.49145975e-03,
        1.57451039e-03,  1.65929255e-03,  1.74558573e-03,  1.83315540e-03,
        1.92175536e-03,  2.01113059e-03,  2.10102008e-03,  2.19115998e-03,
        2.28128654e-03,  2.37113930e-03,  2.46046389e-03,  2.54901493e-03,
        2.63655837e-03,  2.72287393e-03,  2.80775682e-03,  2.89101941e-03,
        2.97249231e-03,  3.05202525e-03,  3.12948745e-03,  3.20476775e-03,
        3.27777434e-03,  3.34843429e-03,  3.41669274e-03,  3.48251197e-03,
        3.54587026e-03,  3.60676063e-03,  3.66518953e-03,  3.77465720e-03,
        3.87470382e-03,  3.96573501e-03,  4.04824552e-03,  4.12278582e-03,
        4.18993448e-03,  4.25027620e-03,  4.30438517e-03,  4.35281300e-03,
        4.43413701e-03,  4.49877661e-03,  4.54984725e-03,  4.58992123e-03,
        4.62109105e-03,  4.66174781e-03,  4.68345968e-03,  4.69228515e-03,
        4.69222005e-03,  4.67352476e-03,  4.64169709e-03,  4.60230907e-03,
        4.55846305e-03,  4.51201539e-03,  4.46414701e-03,  4.41564386e-03,
        4.36705676e-03,  4.27146619e-03,  4.17866236e-03,  4.08944618e-03,
        4.00418521e-03,  3.92299672e-03,  3.84585797e-03,  3.77267182e-03,
        3.63891354e-03,  3.51786146e-03,  3.40820558e-03,  3.30877108e-03,
        3.13973812e-03,  2.99663987e-03,  2.87456569e-03,  2.76983998e-03,
        2.60698555e-03,  2.47957597e-03,  2.37835366e-03,  2.29691410e-03,
        2.23070790e-03,  2.17642314e-03,  2.13159969e-03,  2.09438917e-03,
        2.06343430e-03,  2.01809753e-03,  1.98501094e-03,  1.96076167e-03,
        1.94366484e-03,  1.94366484e-03], dtype=np.float32)
