package ai.demo.translator;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import static ai.demo.translator.App.OUT;

public class Settings
{
    public static final String ATTENTION_GLOBAL = "global";
    public static final String ATTENTION_LOCAL = "local";
    public static final String ATTENTION_NONE = "none";

    private final String path;
    private final int maxLength;
    private final int tokenCount;
    private final int startOfTextToken;
    private final int endOfTextToken;
    private final int contextSize;
    private final int hiddenSize;
    private final int encoderCount;
    private final int decoderCount;
    private final int headCount;
    private final int scoreDividend;
    private final String[] attentionType;
    private final float epsilon;
    private final int localAttentionSize;
    private final boolean hasAttentionQueryBias;
    private final boolean hasAttentionKeyBias;
    private final boolean hasAttentionValueBias;
    private final boolean hasAttentionProjectionBias;
    private final boolean hasMlpLayer1Bias;
    private final boolean hasMlpLayer2Bias;

    public Settings(String path, int maxLength) throws Exception
    {
        this.path = path;
        this.maxLength = maxLength;

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
                        properties.put(parts[0].toLowerCase().trim(), parts[1].toLowerCase().trim());
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
        contextSize = getIntProperty(properties, "context.size");
        hiddenSize = getIntProperty(properties, "hidden.size");
        encoderCount = getIntProperty(properties, "encoder.count");
        decoderCount = getIntProperty(properties, "decoder.count");
        headCount = getIntProperty(properties, "attention.head.count");
        scoreDividend = getIntProperty(properties, "attention.score.dividend");
        epsilon = getFloatProperty(properties, "epsilon");
        hasAttentionQueryBias = getBooleanProperty(properties, "has.attention.query.bias", true);
        hasAttentionKeyBias = getBooleanProperty(properties, "has.attention.key.bias", true);
        hasAttentionValueBias = getBooleanProperty(properties, "has.attention.value.bias", true);
        hasAttentionProjectionBias = getBooleanProperty(properties, "has.attention.projection.bias", true);
        hasMlpLayer1Bias = getBooleanProperty(properties, "has.mlp.layer.1.bias", true);
        hasMlpLayer2Bias = getBooleanProperty(properties, "has.mlp.layer.2.bias", true);

        boolean isLocalUsed = false;

        // Collect the attention type for all decoders
        String[] attentionType = new String[decoderCount];
        for (int i = 0; i < decoderCount; i++)
        {
            String type = getProperty(properties, "attention.type." + (i + 1));

            if (!type.equals(ATTENTION_GLOBAL) && !type.equals(ATTENTION_LOCAL) && !type.equals(ATTENTION_NONE))
            {
                throw new Exception("Incorrect attention type: '" + type + "'" + " Possible values: '"
                        + ATTENTION_GLOBAL + "'," + "'" + ATTENTION_LOCAL + "'," + "'" + ATTENTION_NONE + "'.");
            }

            if (type.equals(ATTENTION_LOCAL)) isLocalUsed = true;

            attentionType[i] = type;
        }
        this.attentionType = attentionType;

        if (isLocalUsed) localAttentionSize = toInt(getProperty(properties, "attention.local.size"));
        else localAttentionSize = Integer.MAX_VALUE;
    }

    public long getParameterSize()
    {
        long wteSize = (long) tokenCount * hiddenSize;
        long wpeSize = (long) contextSize * hiddenSize;
        long finalNormSize = (long) hiddenSize * 2;

        return wteSize + wpeSize + (getDecoderParameterSize() * decoderCount) + finalNormSize;
    }

    private long getDecoderParameterSize()
    {
        long qkvSize = ((long) hiddenSize * hiddenSize + hiddenSize) * 3;
        long projSize = (long) hiddenSize * hiddenSize + hiddenSize;
        long normSize = (long) hiddenSize * 4;
        long layer1Size = ((long) hiddenSize * hiddenSize + hiddenSize) * 4;
        long layer2Size = (long) hiddenSize * hiddenSize * 4 + hiddenSize;

        return qkvSize + projSize + normSize + layer1Size + layer2Size;
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

    public int getMaxLength()
    {
        return maxLength;
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

    public int getDecoderCount()
    {
        return decoderCount;
    }

    public int getHeadCount()
    {
        return headCount;
    }

    public int getScoreDividend()
    {
        return scoreDividend;
    }

    public float getEpsilon()
    {
        return epsilon;
    }

    public String[] getAttentionType()
    {
        return attentionType;
    }

    public int getLocalAttentionSize()
    {
        return localAttentionSize;
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
