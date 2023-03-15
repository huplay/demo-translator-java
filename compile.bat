javac -d target src/main/java/ai/demo/gpt/*.java
cd target
jar cf demo-gpt-java-1.0.jar ai/demo/gpt/*.class
cd ..