# Demo translator # Java

This is a demo application which implements an encoder-decoder Transformer architecture for translation in Java, for learning purposes.

The goal is to demonstrate the encoder-decoder transformer architecture (without training), not to create an optimized application. 

TensorFlow or similar tools are NOT used, everything is implemented here.

## Models ##

| Name                                       | Hidden size | Enc / Dec. no. | Head no. | Length | Size of params |                                                              |
|--------------------------------------------|------------:|---------------:|---------:|-------:|---------------:|--------------------------------------------------------------|
| BART-NYTK-EN-HU <br /> English - Hungarian |         768 |          6 + 6 |       12 |   1024 |          132 M | [Link](https://github.com/huplay/TRANSLATOR-BART-NYTK-EN-HU) | 

## Install ##

1. Install Java (version 1.8 or above).


2. Download and unzip this module: https://github.com/huplay/demo-translator-java

   (Or using git: ```git clone https://github.com/huplay/demo-translator-java.git```)


3. Download and unzip the files with the trained parameters for the version you want to use.


4. Using a command line tool (`cmd`) enter into the main directory:
   
    ```cd demo-translator-java```


5. Compile (build) the application:

   ```compile``` (On Windows)

   Or alternatively (after Maven install): ```mvn clean install```

## Execution ##

Execute the application:
```run < path-of-the-parameters >``` (On Windows)
    
Or on any systems:```java -cp target/demo-translator-java-1.0.jar ai.demo.translator.App < path-of-the-parameters >``` 

To quit press Ctrl + C.

Using larger models it is necessary to increase the heap size (memory for Java). The ```run.bat``` handles it automatically, but if the app is called directly you should use the Java -Xmx and Xms flags. 


## Trained parameters ##

The file format is the simplest you can imagine: the `.dat` files contain the series of big endian float values (4 bytes each).

if a file is too large to upload to GitHub, it is possible to split into multiple files (.part1, .part2, ...), merged automatically when the parameters are read.

The files are placed into folders:
 - `encoders/`
   - `encoder1..n`: Parameter files for the actual encoder  
   - `input`: Parameter files used at the input side of the encoder stack (position embedding, normalization)
 - `decoders/`
   - `decoder1..n`: Parameter files for the actual decoder
   - `input`: Parameter files used at the input side of the decoder stack (position embedding, normalization) 
 - `input`: Parameter files used at the input side of both the encoder and decoder stack (token embedding) 
 - `tokenizer`: Files used by the tokenizer
 - `setup`: It contains command files to set the necessary memory size for the actual model

Every dataset should contain a model.properties file, with the following entries:
 - `token.count`: number of tokens
 - `start.of.text.token`: token id for marking the START-OF-TEXT
 - `end.of.text.token`: token id for marking the END-OF-TEXT
 - `special.token.offset`: number of special tokens with extra embeddings
 - `context.size`: number of tokens the system can process (limited by the position embedding)
 - `hidden.size`: the size of the hidden state
 - `encoder.count`: number of encoders
 - `encoder.attention.head.count`: number of attention heads of the encoders
 - `encoder.attention.score.dividend`: dividend at attention scoring (usually the square root of embeddingSize / headCount)
 - `decoder.count`: number of decoders
 - `decoder.attention.head.count`: number of attention heads of the decoders
 - `decoder.attention.score.dividend`: dividend at attention scoring (usually the square root of embeddingSize / headCount)
 - `epsilon`: epsilon, used at normalization (mostly 1e-5f)
 - `prompt`: the text of the prompt (kind of internationalization)

Optional properties:
 - `name`: name of the model
 - `source`: original creator of the model
 - `hasAttentionQueryBias`: is there a bias for the attention query (default: true)
 - `hasAttentionKeyBias`: is there a bias for the attention key (default: true)
 - `hasAttentionValueBias`: is there a bias for the attention value (default: true)
 - `hasAttentionProjectionBias`: is there a bias for the attention projection (default: true)
 - `hasMlpLayer1Bias`: is there a bias for the mlp layer1 (default: true)
 - `hasMlpLayer2Bias`: is there a bias for the mlp layer2 (default: true)


### Transformer ###

- Attention Is All You Need (2017, Google Brain)
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Usykoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
- https://arxiv.org/abs/1706.03762 
- https://arxiv.org/pdf/1706.03762.pdf

This publication described an encoder-decoder Transformer architecture, optimal for translation between two languages.
The encoder stack creates an inner representation of the input language, the decoder stack transforms this representation to an output in the another language.
(The query, key and value vectors, created by the encoders are passed to the decoders.)
It was implemented using 6 encoders and 6 decoders.


## Read more ##

Jay Alammar: The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/

