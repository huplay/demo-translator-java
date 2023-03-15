package ai.demo.translator;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Converting the input text to list of tokens (encode) and the list of tokens to output text (decode)
 */
public class Tokenizer
{
    private static final String TOKENS_FILENAME = "tokens.map";
    private static final String MERGES_FILENAME = "merges.bpe";

    private final Map<Character, Byte> charEncoding = new HashMap<>(256);
    private final Map<Integer, Character> charDecoding = new HashMap<>(256);

    private final Map<String, Integer> tokenEncoding = new HashMap<>(50257);
    private final Map<Integer, String> tokenDecoding = new HashMap<>(50257);

    private final Map<Pair, Integer> merges = new HashMap<>(50000); // bpe_ranks

    private final Pattern pattern =
            Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");

    /**
     * Initialization
     */
    public Tokenizer(String parametersPath)
    {
        addCharRange(0, 'Ā', 'Ġ');
        addCharRange(33, '!', '~');
        addCharRange(127, 'ġ', 'ł');
        addCharRange(161, '¡', '¬');
        addCharRange(173, 'Ń', 'Ń');
        addCharRange(174, '®', 'ÿ');

        String path = parametersPath + "/tokenizer";

        readTokensFile(path);
        readMergesFile(path);
    }

    private void addCharRange(int pos, char firstChar, char lastChar)
    {
        for (int i = firstChar; i <= lastChar; i++)
        {
            charEncoding.put((char) i, (byte)pos);
            charDecoding.put(pos, (char) i);
            pos++;
        }
    }

    /**
     * Read the "tokens.map" file
     */
    private void readTokensFile(String path)
    {
        String fileName = path + "/" + TOKENS_FILENAME;
        try (Scanner scanner = new Scanner(new File(fileName)))
        {
            while (scanner.hasNext())
            {
                String first = scanner.next();

                if (first.startsWith("\"")) first = first.substring(1);
                if (first.endsWith(":")) first = first.substring(0, first.length() - 1);
                if (first.endsWith("\"")) first = first.substring(0, first.length() - 1);

                first = first.replace("\\\"", "\"");
                first = first.replace("\\'", "'");
                first = first.replace("\\\\", "\\");

                while (true)
                {
                    int i = first.indexOf("\\u");
                    if (i == -1) break;

                    String hex = first.substring(i + 2, i + 6);
                    first = first.replace("\\u" + hex, "" + (char)Integer.parseInt(hex, 16));
                }

                String second = scanner.next();

                if (second.endsWith(",")) second = second.substring(0, second.length() - 1);

                int value = Integer.parseInt(second);

                tokenDecoding.put(value, first);
                tokenEncoding.put(first, value);
            }
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    /**
     * Read the "merges.bpe" file
     */
    private void readMergesFile(String path)
    {
        try
        {
            String fileName = path + "/" + MERGES_FILENAME;
            File file = new File(fileName);
            FileInputStream inputStream = new FileInputStream(file);
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8));

            int i = 0;
            while (true)
            {
                String line = reader.readLine();

                if (line == null) break;

                String[] pairs = line.split(" ");
                merges.put(new Pair(pairs[0], pairs[1]), i);

                i++;
            }

            reader.close();
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    /**
     * Convert text to list of tokens
     */
    public List<Integer> encode(String text)
    {
        if (text == null) return Collections.singletonList(0);

        List<Integer> result = new ArrayList<>();

        Matcher matcher = pattern.matcher(text);
        List<String> unicodes = new ArrayList<>();

        while (matcher.find())
        {
            StringBuilder match = new StringBuilder();

            ByteBuffer buffer = StandardCharsets.UTF_8.encode(matcher.group());
            while (buffer.hasRemaining())
            {
                int value = buffer.get();
                if (value < 0) value = value & 0xff;
                match.append(charDecoding.get(value));
            }

            unicodes.add(match.toString());
        }

        for (String word : unicodes)
        {
            for (String token : bpe(word).split(" "))
            {
                Integer value = tokenEncoding.get(token);
                if (value != null)
                {
                    result.add(value);
                }
            }
        }

        return result;
    }

    /**
     * Convert list of tokens to text
     */
    public String decode(List<Integer> tokens)
    {
        StringBuilder textBuilder = new StringBuilder();
        for (int token : tokens)
        {
            textBuilder.append(tokenDecoding.get(token));
        }
        String text = textBuilder.toString();

        byte[] bytes = new byte[text.length()];
        for (int i = 0; i < text.length(); i++)
        {
            bytes[i] = charEncoding.get(text.charAt(i));
        }

        return new String(bytes, StandardCharsets.UTF_8);
    }

    /**
     * Byte pair encoding
     */
    public String bpe(String token)
    {
        if (token == null || token.length() < 2) return token;

        List<String> word = new ArrayList<>();
        for (char c : token.toCharArray())
        {
            word.add(String.valueOf(c));
        }

        List<Pair> pairs = getPairs(word);

        while (true)
        {
            Pair pair = findFirstPair(pairs);
            if (pair == null) break;

            List<String> newWord = new ArrayList<>();

            int i = 0;
            while (i < word.size())
            {
                int j = findFromIndex(word, pair.left, i);

                if (j != -1)
                {
                    newWord.addAll(word.subList(i, j));
                    i = j;
                }
                else
                {
                    newWord.addAll(word.subList(i, word.size()));
                    break;
                }

                if (word.get(i).equals(pair.left) && i < word.size() - 1 && word.get(i + 1).equals(pair.right))
                {
                    newWord.add(pair.left + pair.right);
                    i = i + 2;
                }
                else
                {
                    newWord.add(word.get(i));
                    i++;
                }
            }

            word = newWord;

            if (word.size() == 1)
            {
                break;
            }
            else
            {
                pairs = getPairs(word);
            }
        }

        return String.join(" ", word);
    }

    /**
     * Split a text into merge pairs
     */
    private List<Pair> getPairs(List<String> word)
    {
        List<Pair> pairs = new ArrayList<>();

        String prev = word.get(0);

        for (String character : word.subList(1, word.size()))
        {
            pairs.add(new Pair(prev, character));
            prev = character;
        }

        return pairs;
    }

    /**
     * Find a String in a list starting from a provided position (from index)
     */
    private int findFromIndex(List<String> input, String find, int from)
    {
        for (int i = from; i < input.size(); i++)
        {
            if (input.get(i).equals(find)) return i;
        }

        return -1;
    }

    /**
     * Find a pair in the merges
     */
    public Pair findFirstPair(List<Pair> pairs)
    {
        int min = Integer.MAX_VALUE;
        Pair minPair = null;

        for (Pair pair : pairs)
        {
            Integer value = merges.get(pair);

            if (value != null && value.compareTo(min) < 0)
            {
                min = value;
                minPair = pair;
            }
        }

        return minPair;
    }

    /**
     * Holder of two string values (pair)
     */
    private static class Pair
    {
        public final String left;
        public final String right;

        public Pair(String left, String right)
        {
            this.left = left;
            this.right = right;
        }

        @Override
        public boolean equals(Object o)
        {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Pair pair = (Pair) o;
            return Objects.equals(left, pair.left) && Objects.equals(right, pair.right);
        }

        @Override
        public int hashCode()
        {
            return Objects.hash(left, right);
        }
    }
}
