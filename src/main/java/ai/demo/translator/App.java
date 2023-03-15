package ai.demo.translator;

import java.io.*;
import java.util.Collections;
import java.util.List;

public class App
{
    public static PrintStream OUT;

    public static void main(String... args) throws Exception
    {
        OUT = new PrintStream(System.out, true, "utf-8");

        OUT.println("________                            __                               __          __");
        OUT.println("\\______ \\   ____   _____   ____   _/  |_____________    ____   _____|  | _____ _/  |_ ___________");
        OUT.println(" |    |  \\_/ __ \\ /     \\ /  _ \\  \\   __\\_  __ \\__  \\  /    \\ /  ___|  | \\__  \\\\   __/  _ \\_  __ \\");
        OUT.println(" |    |   \\  ___/|  Y Y  (  <_> )  |  |  |  | \\// __ \\|   |  \\\\___ \\|  |__/ __ \\|  |(  <_> |  | \\/");
        OUT.println("/_________/\\_____|__|_|__/\\____/   |__|  |__|  (______|___|__/______|____(______|__| \\____/|__|\n");

        try
        {
            if (args == null || args.length == 0)
            {
                throw new Exception("The first parameter should be the path of the model parameters.");
            }

            String path = args[0];

            OUT.println("Path: " + path);

            Settings settings = new Settings(path);

            OUT.println("Number of parameters: " + Math.round(settings.getParameterSize() / 1000000d) + " M");

            OUT.print("\nLoading trained parameters... ");
            Tokenizer tokenizer = new Tokenizer(path);
            Transformer transformer = new Transformer(settings, tokenizer);
            OUT.print("Done.");

            while (true)
            {
                // Read the input text
                OUT.print("\n\nEnglish text: ");
                BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
                String input = reader.readLine();

                // Split the input text into tokens
                List<Integer> inputTokens = tokenizer.encode(input);

                // Use the Transformer
                List<Integer> outputTokens = transformer.processTokens(inputTokens);

                // Convert the output to text and print it
                String response = tokenizer.decode(outputTokens);
                print(response, outputTokens, tokenizer);
            }
        }
        catch (Exception e)
        {
            OUT.println("\nERROR: " + e.getMessage());
        }
    }

    private static void print(String response, List<Integer> outputTokens, Tokenizer tokenizer)
    {
        // The response was printed token by token, but for multi-token characters only "�" will be displayed

        // Here we recreate the token by token decoded response (which wasn't returned)
        StringBuilder tokenByTokenResponse = new StringBuilder();
        for (int token: outputTokens)
        {
            tokenByTokenResponse.append(tokenizer.decode(Collections.singletonList(token)));
        }

        // If the token by token decoded result is different to the final decoded result, print the corrected version
        if ( ! tokenByTokenResponse.toString().equals(response))
        {
            OUT.print("\nCorrected unicode response:\n" + response);
        }
    }
}
