@echo off
REM Point the pipeline at cleaned APTOS 2019 data.
REM
REM Run prepare_aptos.py FIRST -- it repairs missing file extensions, removes
REM duplicate images, and adds the patient_id column this expects:
REM
REM     python prepare_aptos.py --raw-csv aptos\train.csv ^
REM         --images aptos\train_images --out aptos\labels_clean.csv

set DR_CSV=aptos\labels_clean.csv
set DR_IMAGES=aptos\train_images
set DR_CACHE=aptos\cache_512
set DR_FILENAME_COL=filename
set DR_GRADE_COL=diagnosis
set DR_PATIENT_COL=patient_id
set DR_PATIENT_SOURCE=column

set DR_NUM_WORKERS=0
set DR_EPOCHS=30
set DR_BATCH_SIZE=8

echo APTOS environment set. CSV=%DR_CSV%
echo.
echo On a CPU-only machine, training this will take many hours.
echo Consider Google Colab: same commands, free GPU.
echo.
echo Next:  python check_setup.py
