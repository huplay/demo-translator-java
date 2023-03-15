@echo off
CHCP 65001 >nul
if exist %1/setup/setup-java.bat call %1/setup/setup-java.bat
java %GPT_JAVA_ARGS% -cp target/demo-gpt-java-1.0.jar ai.demo.gpt.App %*

