# XGBoost nowcasting for SMR and Pc5 ULF


XGBoost models for nowcasting SuperMAG ring current and Pc5 ultra low frequency indices. This is the code for my statistics master's thesis at the University of Helsinki. Filepaths have been emptied so if you intend to use this code, you need to fill your own filepaths. Python version 3.12.3 was used. 


There might be some inconsistencies on data loading date formats. However, all necessary strings are present either in the right place or as comment on the same line. In addition, as I can't provide the preprocessed data due to SuperMAG policy, it may be that hypothetical other developers end up loading the data differently. Another possible source of confusion is the use of load_preprocessed_data() function. I ended up using it to load a LazyFrame but the code may somewhere assume that it loads a DataFrame instead. Just add .collect() and you get the DataFrame.
