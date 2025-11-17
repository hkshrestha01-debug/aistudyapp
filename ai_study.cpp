// ai_study.cpp
// Simple terminal AI study assistant that:
// - Calls OpenAI's Chat Completions API via libcurl
// - Summarizes pasted study text
// - Generates flashcards and lets you flip through them in a terminal UI

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <stdexcept>
#include <limits>
#include <random>
#include <algorithm>

#include <curl/curl.h>          // HTTP requests to OpenAI
#include <nlohmann/json.hpp>    // JSON parsing (https://github.com/nlohmann/json)

using json = nlohmann::json;

// ======== DATA STRUCTS =========

// Holds a single term + its definition
struct Definition {
    std::string term;
    std::string definition;
};

// Result object for a summary request
struct SummaryResult {
    std::string summary;                 // main summary text
    std::vector<std::string> keyPoints;  // bullet key points
    std::vector<Definition> definitions; // list of definitions found in the text
};

// Represents a single flashcard
struct Flashcard {
    std::string question;
    std::string answer;
};

// Result object for flashcard generation
struct FlashcardResult {
    std::vector<Flashcard> flashcards;
};

// ======== HELPER TO EXTRACT JSON FROM MODEL REPLY =========

// Takes the assistant message content (which might include markdown, text, etc.)
// and extracts the JSON object between the first '{' and the last '}'.
static std::string extract_json_block(const std::string& content) {
    auto firstBrace = content.find('{');
    auto lastBrace  = content.rfind('}');

    if (firstBrace == std::string::npos ||
        lastBrace  == std::string::npos ||
        lastBrace <= firstBrace) {
        throw std::runtime_error(
            "Assistant response did not contain a valid JSON object:\n" + content
        );
    }

    return content.substr(firstBrace, lastBrace - firstBrace + 1);
}

// ======== TERMINAL UI HELPERS =========

// Clears the terminal screen using ANSI escape codes
static void clear_screen() {
    std::cout << "\033[2J\033[H";
}

// Renders a single flashcard (and optionally the answer) to the terminal
static void display_card(const Flashcard& card, int index, int total, bool showAnswer) {
    clear_screen();
    std::cout << "Flashcard " << (index + 1) << "/" << total << "\n";
    std::cout << "-------------------------\n";
    std::cout << "Q: " << card.question << "\n\n";
    if (showAnswer) {
        std::cout << "A: " << card.answer << "\n\n";
    } else {
        std::cout << "A: [hidden] (press 'f' to flip)\n\n";
    }
    std::cout << "Commands: [f]lip  [n]ext  [p]rev  [r]andom  [j]ump <num>  [q]uit\n";
}

// Interactive flashcard viewer loop for the terminal
static void run_flashcard_viewer(const FlashcardResult& deck) {
    // If no flashcards, just exit
    if (deck.flashcards.empty()) {
        std::cout << "No flashcards to view.\n";
        return;
    }

    int idx = 0;                      // current flashcard index
    bool showAnswer = false;          // whether answer is visible
    std::string cmd;                  // user command/input line
    std::mt19937 rng((unsigned)std::random_device{}()); // RNG for random card

    while (true) {
        // Display current card
        display_card(deck.flashcards[idx], idx, (int)deck.flashcards.size(), showAnswer);

        // Read a command line from user
        if (!std::getline(std::cin, cmd)) break; // if EOF, exit
        if (cmd.empty()) continue;               // ignore empty lines

        // Trim leading spaces
        size_t p = cmd.find_first_not_of(" \t");
        if (p != std::string::npos) cmd = cmd.substr(p);

        // Handle supported commands
        if (cmd == "f" || cmd == "flip") {
            // Toggle answer visibility
            showAnswer = !showAnswer;

        } else if (cmd == "n" || cmd == "next") {
            // Move to next card (wrap around)
            idx = (idx + 1) % (int)deck.flashcards.size();
            showAnswer = false;

        } else if (cmd == "p" || cmd == "prev") {
            // Move to previous card (wrap around)
            idx = (idx - 1 + (int)deck.flashcards.size()) % (int)deck.flashcards.size();
            showAnswer = false;

        } else if (cmd == "r" || cmd == "random") {
            // Jump to random card
            std::uniform_int_distribution<int> dist(0, (int)deck.flashcards.size() - 1);
            idx = dist(rng);
            showAnswer = false;

        } else if (cmd.size() > 2 && (cmd[0] == 'j' || cmd.rfind("jump", 0) == 0)) {
            // "jump" command (e.g., "j 3" or "jump 5")
            std::string numstr;
            // Extract digits and optional minus sign from the string
            for (char c : cmd)
                if ((c >= '0' && c <= '9') || c == '-')
                    numstr.push_back(c);

            if (!numstr.empty()) {
                try {
                    int t = std::stoi(numstr);
                    // Only jump if index is in valid range
                    if (t >= 1 && t <= (int)deck.flashcards.size()) {
                        idx = t - 1;
                        showAnswer = false;
                    }
                } catch (...) {
                    // Ignore invalid numbers
                }
            }

        } else if (cmd == "q" || cmd == "quit") {
            // Quit viewer
            break;

        } else {
            // If the command isn't recognized, try to interpret it as a card number
            try {
                int t = std::stoi(cmd);
                if (t >= 1 && t <= (int)deck.flashcards.size()) {
                    idx = t - 1;
                    showAnswer = false;
                }
            } catch (...) {
                // Unknown input, ignore
            }
        }
    }
    clear_screen();
}

// ======== CURL RESPONSE CALLBACK =========

// Callback that libcurl uses to write incoming HTTP response data into a std::string
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t totalSize = size * nmemb;
    std::string* s = static_cast<std::string*>(userp);
    s->append(static_cast<char*>(contents), totalSize);
    return totalSize;
}

// ======== CORE OPENAI CALLER =========

// Sends a prompt to OpenAI Chat Completions API and returns the raw JSON response as a string
std::string call_openai_chat(const std::string& prompt) {
    // Grab API key from environment variable
    const char* envKey = std::getenv("OPENAI_API_KEY");
    if (!envKey) {
        throw std::runtime_error("OPENAI_API_KEY environment variable not set.");
    }
    std::string apiKey = envKey;

    // Initialize CURL handle
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to init curl");
    }

    std::string readBuffer;  // will hold full HTTP response

    const char* url = "https://api.openai.com/v1/chat/completions";

    // Build JSON payload to send to OpenAI
    json body;
    body["model"] = "gpt-4.1-mini";    // model name
    body["messages"] = {               // single user message with prompt
        {
            {"role", "user"},
            {"content", prompt}
        }
    };
    std::string bodyStr = body.dump(); // serialize JSON to string

    // Set HTTP headers (JSON + Authorization)
    struct curl_slist* headers = nullptr;
    std::string authHeader = "Authorization: Bearer " + apiKey;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, authHeader.c_str());

    // Configure CURL options
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, bodyStr.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback); // callback for incoming data
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);       // store data in readBuffer

    // Perform the HTTP POST
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        throw std::runtime_error(std::string("curl_easy_perform() failed: ") +
                                 curl_easy_strerror(res));
    }

    // Check HTTP status code
    long httpCode = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
    if (httpCode < 200 || httpCode >= 300) {
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        throw std::runtime_error("OpenAI API returned HTTP code " +
                                 std::to_string(httpCode) +
                                 "\nResponse: " + readBuffer);
    }

    // Clean up headers and CURL handle
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    // Return raw JSON response string
    return readBuffer;
}

// ======== AI LOGIC: SUMMARY =========

// Sends text to OpenAI with a prompt asking for:
// - summary
// - key points
// - definitions
// and parses the JSON result into SummaryResult
SummaryResult summarize_content(const std::string& text) {
    // Prompt instructing the model to reply ONLY with JSON in a specific shape
    std::string prompt = R"(
You are an AI study assistant.

TASK:
1. Read the following text.
2. Write a concise summary (150–250 words) in simple language.
3. List 3–5 key points.
4. If there are definitions, include them in your own words.

Return ONLY valid JSON with this structure:
{
  "summary": "string",
  "key_points": ["string", "string"],
  "definitions": [
    {"term": "string", "definition": "string"}
  ]
}

TEXT:
)";
    // Append user-pasted text after the prompt
    prompt += text;

    // Call OpenAI
    std::string rawResponse = call_openai_chat(prompt);

    // Parse top-level API response JSON
    json resJson = json::parse(rawResponse);

    // Extract the assistant's message content
    std::string content;
    const auto& msgContent = resJson["choices"][0]["message"]["content"];

    if (msgContent.is_string()) {
        content = msgContent.get<std::string>();
    } else if (msgContent.is_array()) {
        // In case the API returns content as an array of parts
        for (const auto& part : msgContent) {
            if (part.contains("text") && part["text"].is_string()) {
                content += part["text"].get<std::string>();
            }
        }
    } else {
        throw std::runtime_error("Unexpected content format in OpenAI response.");
    }

    // Extract pure JSON block from the content (removes ```json fences, text, etc.)
    std::string jsonText = extract_json_block(content);

    // Parse the assistant message content as JSON
    json summaryJson = json::parse(jsonText);

    // Fill in the SummaryResult struct
    SummaryResult result;
    result.summary = summaryJson.value("summary", "");

    // Key points list
    if (summaryJson.contains("key_points") && summaryJson["key_points"].is_array()) {
        for (auto& kp : summaryJson["key_points"]) {
            result.keyPoints.push_back(kp.get<std::string>());
        }
    }

    // Definitions list
    if (summaryJson.contains("definitions") && summaryJson["definitions"].is_array()) {
        for (auto& d : summaryJson["definitions"]) {
            Definition def;
            def.term = d.value("term", "");
            def.definition = d.value("definition", "");
            result.definitions.push_back(def);
        }
    }

    return result;
}

// ======== AI LOGIC: FLASHCARDS =========

// Sends text to OpenAI asking it to generate a JSON list of flashcards
FlashcardResult generate_flashcards(const std::string& text) {
    // Prompt instructing the model on how to generate flashcards
    std::string prompt = R"(
You are an AI that creates study flashcards.

Given the TEXT below, create 10–20 flashcards that help a student study.

Rules:
- Questions should be clear and specific.
- Answers should be brief (1–3 sentences).
- Mix definitions, concepts, and reasoning questions.

Return ONLY valid JSON with this structure:
{
  "flashcards": [
    {"question": "string", "answer": "string"}
  ]
}

TEXT:
)";
    // Attach study text to the prompt
    prompt += text;

    // Call OpenAI and parse
    std::string rawResponse = call_openai_chat(prompt);

    json resJson = json::parse(rawResponse);

    // Extract the assistant's message content
    std::string content;
    const auto& msgContent = resJson["choices"][0]["message"]["content"];

    if (msgContent.is_string()) {
        content = msgContent.get<std::string>();
    } else if (msgContent.is_array()) {
        for (const auto& part : msgContent) {
            if (part.contains("text") && part["text"].is_string()) {
                content += part["text"].get<std::string>();
            }
        }
    } else {
        throw std::runtime_error("Unexpected content format in OpenAI response.");
    }

    // Extract and parse the JSON block
    std::string jsonText = extract_json_block(content);
    json fcJson = json::parse(jsonText);

    FlashcardResult result;
    // Extract flashcards from JSON array
    if (fcJson.contains("flashcards") && fcJson["flashcards"].is_array()) {
        for (auto& fc : fcJson["flashcards"]) {
            Flashcard card;
            card.question = fc.value("question", "");
            card.answer   = fc.value("answer", "");
            result.flashcards.push_back(card);
        }
    }

    return result;
}

// ======== DEMO MAIN =========

int main() {
    // Global initialization for libcurl (must be paired with curl_global_cleanup)
    curl_global_init(CURL_GLOBAL_DEFAULT);

    try {
        // 1) Ask user what they want the app to do
        std::cout << "What do you want?\n";
        std::cout << "1 = Summary only\n";
        std::cout << "2 = Flashcards only\n";
        std::cout << "3 = Both summary + flashcards\n";
        std::cout << "Enter choice (1/2/3): ";

        int choice = 3;  // default to "both" if user input fails
        std::cin >> choice;
        // Clear leftover newline from the input buffer before using getline()
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        std::string userText;

        {
            // 2) Read multi-line study text from the user
            std::cout << "\nPaste your study text below.\n";
            std::cout << "When you're done, press Enter on an empty line to finish input.\n";
            std::cout << "Then press Enter.\n\n";

            std::string line;

            // Read the first line
            if (!std::getline(std::cin, line)) {
                std::cerr << "No input detected. Exiting.\n";
                curl_global_cleanup();
                return 0;
            }

            if (line.empty()) {
                // If the very first line is empty, treat as no input
                std::cerr << "No text entered. Exiting.\n";
                curl_global_cleanup();
                return 0;
            }

            // Start building userText with the first line
            userText += line;

            // If the user ends a line with a backslash '\',
            // keep reading additional lines and append them.
            // This is a manual "multiline" mode.
            while (!userText.empty() && userText.back() == '\\') {
                // Remove the trailing backslash and add a newline
                userText.pop_back();
                userText += '\n';

                if (!std::getline(std::cin, line)) break;

                // Stop if line is empty (user pressed Enter)
                if (line.empty()) break;

                // Append the newly read line
                userText += line;
            }

            // Final check: if userText ended up empty, stop
            if (userText.empty()) {
                std::cerr << "No text entered. Exiting.\n";
                curl_global_cleanup();
                return 0;
            }
        }

        // 3) Based on user choice, call summary and/or flashcard functions

        // SUMMARY FLOW
        if (choice == 1 || choice == 3) {
            SummaryResult s = summarize_content(userText);

            std::cout << "\n=== SUMMARY ===\n" << s.summary << "\n\n";

            std::cout << "Key points:\n";
            for (const auto& kp : s.keyPoints) {
                std::cout << "- " << kp << "\n";
            }

            std::cout << "\nDefinitions:\n";
            for (const auto& d : s.definitions) {
                std::cout << d.term << ": " << d.definition << "\n";
            }
        }

        // FLASHCARD FLOW
        if (choice == 2 || choice == 3) {
            FlashcardResult f = generate_flashcards(userText);
            // Launch interactive viewer only if we actually have flashcards
            run_flashcard_viewer(f);
        }

    } catch (const std::exception& ex) {
        // If any exception happens (curl, JSON, etc.), print error message
        std::cerr << "Error: " << ex.what() << "\n";
    }

    // Clean up global curl state
    curl_global_cleanup();
    return 0;
}
