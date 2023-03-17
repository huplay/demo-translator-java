package ai.demo.translator;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import static ai.demo.translator.App.OUT;

public class Settings
{
    private final String path;

    private final int tokenCount;
    private final int startOfTextToken;
    private final int endOfTextToken;
    private final int specialTokenOffset;

    private final int contextSize;
    private final int hiddenSize;

    private final int encoderCount;
    private final int encoderHeadCount;
    private final int encoderScoreDividend;

    private final int decoderCount;
    private final int decoderHeadCount;
    private final int decoderScoreDividend;

    private final float epsilon;

    private final String prompt;

    private final boolean hasAttentionQueryBias;
    private final boolean hasAttentionKeyBias;
    private final boolean hasAttentionValueBias;
    private final boolean hasAttentionProjectionBias;
    private final boolean hasMlpLayer1Bias;
    private final boolean hasMlpLayer2Bias;

    public Settings(String path) throws Exception
    {
        this.path = path;

        // Read all properties from the model.properties file
        Map<String, String> properties = new HashMap<>();

        String fileName = path + "/model.properties";
        try (Scanner scanner = new Scanner(new File(fileName)))
        {
            while (scanner.hasNextLine())
            {
                String line = scanner.nextLine();
                if (line != null && !line.trim().equals("") && !line.startsWith("#"))
                {
                    String[] parts = line.split("=");
                    if (parts.length == 2)
                    {
                        properties.put(parts[0].toLowerCase().trim(), parts[1].trim());
                    }
                    else
                    {
                        OUT.println("\nWARNING: Unrecognizable properties line: (" + fileName + "): " + line);
                    }
                }
            }
        }
        catch (IOException e)
        {
            throw new Exception("Cannot read model.properties file: " + fileName);
        }

        // Find the necessary in the collected properties
        tokenCount = getIntProperty(properties, "token.count");
        startOfTextToken = getIntProperty(properties, "start.of.text.token");
        endOfTextToken = getIntProperty(properties, "end.of.text.token");
        specialTokenOffset = getIntProperty(properties, "special.token.offset");

        contextSize = getIntProperty(properties, "context.size");
        hiddenSize = getIntProperty(properties, "hidden.size");

        encoderCount = getIntProperty(properties, "encoder.count");
        encoderHeadCount = getIntProperty(properties, "encoder.attention.head.count");
        encoderScoreDividend = getIntProperty(properties, "encoder.attention.score.dividend");

        decoderCount = getIntProperty(properties, "decoder.count");
        decoderHeadCount = getIntProperty(properties, "decoder.attention.head.count");
        decoderScoreDividend = getIntProperty(properties, "decoder.attention.score.dividend");

        epsilon = getFloatProperty(properties, "epsilon");

        prompt = getProperty(properties, "prompt");

        hasAttentionQueryBias = getBooleanProperty(properties, "has.attention.query.bias", true);
        hasAttentionKeyBias = getBooleanProperty(properties, "has.attention.key.bias", true);
        hasAttentionValueBias = getBooleanProperty(properties, "has.attention.value.bias", true);
        hasAttentionProjectionBias = getBooleanProperty(properties, "has.attention.projection.bias", true);
        hasMlpLayer1Bias = getBooleanProperty(properties, "has.mlp.layer.1.bias", true);
        hasMlpLayer2Bias = getBooleanProperty(properties, "has.mlp.layer.2.bias", true);
    }

    public long getParameterSize()
    {
        long wteSize = (long) tokenCount * hiddenSize;

        long wpeSize = (long) (contextSize + specialTokenOffset) * hiddenSize;
        long finalNormSize = (long) hiddenSize * 2;

        return wteSize +
                wpeSize + (getEncoderParameterSize() * encoderCount) + finalNormSize +
                wpeSize + (getDecoderParameterSize() * decoderCount) + finalNormSize;
    }

    private long getEncoderParameterSize()
    {
        long qkvSize = ((long) hiddenSize * hiddenSize + hiddenSize) * 3;
        long projSize = (long) hiddenSize * hiddenSize + hiddenSize;
        long normSize = (long) hiddenSize * 4;
        long layer1Size = ((long) hiddenSize * hiddenSize + hiddenSize) * 4;
        long layer2Size = (long) hiddenSize * hiddenSize * 4 + hiddenSize;

        return qkvSize + projSize + normSize + layer1Size + layer2Size;
    }

    private long getDecoderParameterSize()
    {
        long qkvSize = ((long) hiddenSize * hiddenSize + hiddenSize) * 3;
        long projSize = (long) hiddenSize * hiddenSize + hiddenSize;
        long normSize = (long) hiddenSize * 4;
        long layer1Size = ((long) hiddenSize * hiddenSize + hiddenSize) * 4;
        long layer2Size = (long) hiddenSize * hiddenSize * 4 + hiddenSize;

        return 2 * (qkvSize + projSize + normSize) + layer1Size + layer2Size;
    }

    private int getIntProperty(Map<String, String> properties, String key) throws Exception
    {
        return toInt(getProperty(properties, key));
    }

    private float getFloatProperty(Map<String, String> properties, String key) throws Exception
    {
        return toFloat(getProperty(properties, key));
    }

    private boolean getBooleanProperty(Map<String, String> properties, String key, boolean defaultValue) throws Exception
    {
        return toBoolean(getProperty(properties, key, true), defaultValue);
    }

    private String getProperty(Map<String, String> properties, String key) throws Exception
    {
        return getProperty(properties, key, false);
    }

    private String getProperty(Map<String, String> properties, String key, boolean isOptional) throws Exception
    {
        String value = properties.get(key);

        if (!isOptional && value == null)
        {
            throw new Exception("Missing entry in the model.properties file: '" + key + "'.");
        }

        return value;
    }

    private int toInt(String value) throws Exception
    {
        try
        {
            return Integer.parseInt(value);
        }
        catch (Exception e)
        {
            throw new Exception("The provided properties value can't be converted to integer (" + value + ").");
        }
    }

    private float toFloat(String value) throws Exception
    {
        try
        {
            return Float.parseFloat(value);
        }
        catch (Exception e)
        {
            throw new Exception("The provided properties value can't be converted to float (" + value + ").");
        }
    }

    private boolean toBoolean(String value, boolean defaultValue) throws Exception
    {
        if (value == null) return defaultValue;

        try
        {
            return Boolean.parseBoolean(value);
        }
        catch (Exception e)
        {
            throw new Exception("The provided properties value can't be converted to boolean (" + value + ").");
        }
    }

    public String getPath()
    {
        return path;
    }

    public int getTokenCount()
    {
        return tokenCount;
    }

    public int getStartOfTextToken()
    {
        return startOfTextToken;
    }

    public int getEndOfTextToken()
    {
        return endOfTextToken;
    }

    public int getSpecialTokenOffset()
    {
        return specialTokenOffset;
    }

    public int getContextSize()
    {
        return contextSize;
    }

    public int getHiddenSize()
    {
        return hiddenSize;
    }

    public int getEncoderCount()
    {
        return encoderCount;
    }

    public int getEncoderHeadCount()
    {
        return encoderHeadCount;
    }

    public int getEncoderScoreDividend()
    {
        return encoderScoreDividend;
    }

    public int getDecoderCount()
    {
        return decoderCount;
    }

    public int getDecoderHeadCount()
    {
        return decoderHeadCount;
    }

    public int getDecoderScoreDividend()
    {
        return decoderScoreDividend;
    }

    public float getEpsilon()
    {
        return epsilon;
    }

    public String getPrompt()
    {
        return prompt;
    }

    public boolean hasAttentionQueryBias()
    {
        return hasAttentionQueryBias;
    }

    public boolean hasAttentionKeyBias()
    {
        return hasAttentionKeyBias;
    }

    public boolean hasAttentionValueBias()
    {
        return hasAttentionValueBias;
    }

    public boolean hasAttentionProjectionBias()
    {
        return hasAttentionProjectionBias;
    }

    public boolean hasMlpLayer1Bias()
    {
        return hasMlpLayer1Bias;
    }

    public boolean hasMlpLayer2Bias()
    {
        return hasMlpLayer2Bias;
    }
}
