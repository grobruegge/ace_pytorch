rem Source and destination folder paths
set "source_folder=sourceDir2\imagenet_images"
set "destination_base=sourceDir2\random_"

rem Total number of iterations
set "total_iterations=20"

rem Number of files to move in each iteration
set "files_per_iteration=500"

for /l %%i in (0,1,%total_iterations%) do (
    set "destination_folder=!destination_base!%%i"
    mkdir "!destination_folder!" 2>nul 

    rem Use PowerShell to randomly select 500 files from the source folder
    powershell -command "Get-ChildItem -Path '!source_folder!' -File | Get-Random -Count %files_per_iteration% | ForEach-Object { Move-Item -Path $_.FullName -Destination '!destination_folder!' }"

    echo Moved %files_per_iteration% files to !destination_folder!
)

endlocal
