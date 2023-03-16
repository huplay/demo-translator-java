javac -d target src/main/java/ai/demo/translator/*.java
cd target
jar cf demo-translator-java-1.0.jar ai/demo/translator/*.class
cd ..