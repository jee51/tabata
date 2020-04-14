rem for /F "delims= " %i in ('pip list --outdated') do echo %i
for /F "delims= " %%i in ('pip list --outdated') do echo pip install -U %%i