@echo off
REM Point the pipeline at the synthetic demo dataset.
REM
REM Usage (Windows Command Prompt, from the project folder):
REM     demo_env.bat
REM
REM Environment variables set with `set` last only for the current window, so
REM run this again each time you open a new Command Prompt. Values here override
REM the defaults in config.py without editing it.

set DR_CSV=demo_data\labels.csv
set DR_IMAGES=demo_data\images
set DR_CACHE=demo_data\cache_512
set DR_FILENAME_COL=id_code
set DR_GRADE_COL=diagnosis
set DR_PATIENT_COL=patient_id
set DR_PATIENT_SOURCE=column

REM Windows DataLoader workers use spawn rather than fork: slow, and a frequent
REM source of confusing pickle errors. Zero is the right value here.
set DR_NUM_WORKERS=0

REM Small and short so a CPU-only machine finishes. These numbers are for
REM checking that the pipeline runs, not for producing a usable model.
set DR_EPOCHS=3
set DR_BATCH_SIZE=4

echo Demo environment set:
echo   CSV     %DR_CSV%
echo   images  %DR_IMAGES%
echo   cache   %DR_CACHE%
echo   workers %DR_NUM_WORKERS%  epochs %DR_EPOCHS%  batch %DR_BATCH_SIZE%
echo.
echo Next:  python check_setup.py
