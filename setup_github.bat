@echo off
REM This script helps you push to the GitHub repository

echo Setting up GitHub repository for Data Chat...

REM Check if git is installed
git --version > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Git is not installed or not in your PATH.
    echo Please install Git from https://git-scm.com/
    exit /b 1
)

REM Set remote origin to the provided URL
echo Setting remote origin to https://github.com/whereAGI/advanced_data_analysis_csv.git
git remote remove origin 2>nul
git remote add origin https://github.com/whereAGI/advanced_data_analysis_csv.git

REM Add files to repository
echo.
echo Adding files to repository...
git add .

REM Check if there are any existing commits
git rev-parse --verify HEAD >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Creating initial commit...
    git commit -m "Initial commit: Data Chat application"
) else (
    echo Creating new commit...
    git commit -m "Update: Data Chat application"
)

REM Push to GitHub
echo.
echo Pushing to GitHub...
git push -u origin master

if %ERRORLEVEL% neq 0 (
    echo Failed to push to master branch, trying main branch...
    git push -u origin main
    
    if %ERRORLEVEL% neq 0 (
        echo Failed to push. Trying to create branch first...
        git checkout -b master
        git push -u origin master
        
        if %ERRORLEVEL% neq 0 (
            echo Failed to push to GitHub. Please check your credentials and repository URL.
            exit /b 1
        )
    )
)

echo.
echo Success! Your Data Chat application has been pushed to GitHub.
echo.

exit /b 0 