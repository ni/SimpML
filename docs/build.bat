@echo off

:: Make the current directory the directory of this script.
pushd "%~dp0\"

if exist "_build" rd /s /q "_build"
mkdir "_build"
xcopy /s /y /q /r /i "source" "_build\source"
xcopy /s /y /q /r /i  "examples" "_build\source\examples"
sphinx-apidoc -e -o "_build\source\_modules" "..\simpml"
sphinx-build -M html "_build\source" "_build"

:: Change the current directory back to the original directory.
popd
