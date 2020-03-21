@echo off
cd ..
c:
cd Users\%username%\desktop
for /l %%x in (0,1,9) do (
    echo %%x
    mkdir %%x
    robocopy  ./test ./%%x  "%%x_*.*" /MIR /MT:100 /XO
)

pause