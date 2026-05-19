' *******************************************************
' * GPT-2 BASIC Main Program                           *
' *******************************************************
' * This is the main entry point for the GPT-2 BASIC    *
' * implementation. It brings together all the modules  *
' * and provides a text completion application.         *
' *                                                     *
' * This implementation demonstrates a scaled-down      *
' * transformer architecture optimized for 486-era      *
' * hardware with 32MB of RAM.                          *
' *******************************************************

#INCLUDE "src/data_structures.bas"
#INCLUDE "src/matrix_ops.bas"
#INCLUDE "src/simd_ops.bas"
#INCLUDE "src/block_sparse.bas"
#INCLUDE "src/memory_manager.bas"
#INCLUDE "src/asm_optimizations.bas"
#INCLUDE "src/softmax_fixed.bas"
#INCLUDE "src/tokenizer.bas"
#INCLUDE "src/real_gpt.bas"
#INCLUDE "src/quality_prior.bas"
#INCLUDE "src/transformer_components.bas"
#INCLUDE "src/model.bas"
#INCLUDE "src/file_io.bas"
#INCLUDE "src/benchmark.bas"

DECLARE FUNCTION GenerateNextToken(model AS GPT2Model, context() AS INTEGER, context_len AS INTEGER, temperature AS SINGLE, top_p AS SINGLE, top_k AS INTEGER) AS INTEGER
DECLARE FUNCTION ArgMax(logits() AS SINGLE) AS INTEGER
DECLARE SUB TopK(logits() AS SINGLE, k AS INTEGER)
DECLARE SUB TopP(logits() AS SINGLE, p AS SINGLE)
DECLARE FUNCTION SampleFromLogits(logits() AS SINGLE) AS INTEGER
DECLARE FUNCTION GetCPUTypeName() AS STRING
DECLARE SUB PrintActiveModelInfo()
DECLARE FUNCTION PerfLongText(value AS LONG) AS STRING
DECLARE FUNCTION PerfDoubleText(value AS DOUBLE) AS STRING
DECLARE SUB RunHardwarePerformanceSuite()
DECLARE SUB RunHardwarePerformanceCase(case_name AS STRING, prompt AS STRING, max_length AS INTEGER, BYREF total_runs AS INTEGER, BYREF total_tokens AS LONG, BYREF total_seconds AS DOUBLE)

' *******************************************************
' * Constants and Configuration                         *
' *******************************************************

' Default model configuration settings
CONST MODEL_PATH = "MODEL"
CONST VOCAB_PATH = "VOCAB.BIN"
CONST MODEL_VOCAB_PATH = "MODEL\VOCAB.BIN"
CONST CONFIG_PATH = "CONFIG.TXT"

' Default generation parameters
CONST DEFAULT_MAX_LENGTH = 220
CONST DEFAULT_TEMPERATURE = 0.7
CONST DEFAULT_TOP_P = 0.9
CONST DEFAULT_TOP_K = 40
CONST SENTENCE_STOP_MIN_TOKENS = 30
CONST ALLOW_LEGACY_MATRIX_FALLBACK = 0

' *******************************************************
' * Global Variables                                    *
' *******************************************************

DIM SHARED g_model_initialized AS INTEGER

' *******************************************************
' * Utility Functions                                   *
' *******************************************************

' Clear the screen - DOS specific
SUB ClearScreen()
    CLS
END SUB

' Wait for a key press
SUB WaitForKeypress()
    PRINT "Press any key to continue..."
    WHILE INKEY$ = "": WEND
END SUB

' Print a centered title
SUB PrintTitle(title AS STRING, screen_width AS INTEGER)
    DIM pad_length AS INTEGER
    pad_length = (screen_width - LEN(title)) \ 2
    PRINT STRING$(pad_length, " "); title
END SUB

' Print a separator line
SUB PrintSeparator(line_width AS INTEGER, separator_char AS STRING)
    PRINT STRING$(line_width, separator_char)
END SUB

' Print memory usage information
SUB PrintMemoryUsage()
    PRINT "Memory Usage:"
    PRINT "  Current: "; g_current_memory_usage / (1024 * 1024); "MB"
    PRINT "  Peak   : "; g_peak_memory_usage / (1024 * 1024); "MB"
    PRINT "  Available: "; GetAvailableMemory() / (1024 * 1024); "MB"
END SUB

SUB PrintActiveModelInfo()
    IF GPT2BasicIsLoaded() = 0 THEN
        PRINT "  Model       : not loaded"
        RETURN
    END IF

    PRINT "  Profile     : "; GPT2BasicProfileName()
    PRINT "  Layers      : "; GPT2BasicLayerCount()
    PRINT "  Embed dim   : "; GPT2BasicEmbeddingDim()
    PRINT "  Heads       : "; GPT2BasicHeadCount()
    PRINT "  Vocab size  : "; GPT2BasicVocabSize()
    PRINT "  Max tokens  : "; GPT2BasicContextLength()
    PRINT "  Parameters  : "; GPT2BasicParameterCount()
    PRINT "  Fixed bytes : "; GPT2BasicFixedWeightBytes()
    PRINT "  Runtime mem : "; GPT2BasicRuntimeMemoryBytes()
    PRINT "  Arithmetic  : ";
    IF GPT2BasicIsFixedPointLoaded() <> 0 THEN
        PRINT "Q20.12 fixed-point"
    ELSE
        PRINT "float reference"
    END IF
END SUB

SUB StripTrailingEOT(tokens() AS INTEGER, BYREF token_count AS INTEGER)
    IF token_count <= 0 THEN RETURN
    IF tokens(token_count - 1) = 0 THEN token_count = token_count - 1
END SUB

FUNCTION PerfLongText(value AS LONG) AS STRING
    RETURN LTRIM$(STR$(value))
END FUNCTION

FUNCTION PerfDoubleText(value AS DOUBLE) AS STRING
    RETURN LTRIM$(STR$(value))
END FUNCTION

' *******************************************************
' * Model Setup and Initialization                      *
' *******************************************************

' Initialize the model and all components
FUNCTION InitializeModel() AS INTEGER
    DIM success AS INTEGER
    DIM vocab_source AS STRING

    ' Skip if already initialized
    IF g_model_initialized THEN
        RETURN 1
    END IF

    PRINT "Initializing GPT-2 BASIC model..."

    ' Initialize memory manager
    InitMemoryManager()

    ' Initialize matrix operations
    InitMatrixOps()

    ' Initialize assembly optimizations
    InitAsmOptimizations()

    ' Initialize tokenizer
    InitializeDefaultTokenizer()

    ' Attempt to load vocabulary if an external one is present.
    ' The default byte-level tokenizer is already initialized above.
    vocab_source = ""
    IF DIR(VOCAB_PATH) <> "" THEN
        vocab_source = VOCAB_PATH
    ELSEIF DIR(MODEL_VOCAB_PATH) <> "" THEN
        vocab_source = MODEL_VOCAB_PATH
    END IF

    IF vocab_source <> "" THEN
        success = 0
        ON ERROR GOTO vocab_error
        LoadDefaultVocabulary(vocab_source)
        success = 1
        GOTO vocab_done

vocab_error:
        success = 0

vocab_done:
        ON ERROR GOTO 0
        IF success = 0 THEN
            PRINT "Warning: Could not load vocabulary from "; vocab_source
            PRINT "Using built-in byte-level vocabulary."
            InitializeDefaultTokenizer()
        END IF
    ELSE
        PRINT "Using built-in byte-level vocabulary."
    END IF

    ' Load model configuration
    success = LoadTextModelConfig(g_config, CONFIG_PATH)

    IF success = 0 THEN
        PRINT "Warning: Could not load configuration from "; CONFIG_PATH
        PRINT "Using default configuration."

        ' Set default configuration
        InitModelConfig(g_config)
    END IF

    IF GPT2BasicLoadModel(MODEL_PATH) <> 0 THEN
        g_config.n_embd = GPT2BasicEmbeddingDim()
        g_config.n_head = GPT2BasicHeadCount()
        g_config.n_layer = GPT2BasicLayerCount()
        g_config.n_positions = GPT2BasicContextLength()
        g_config.vocab_size = GPT2BasicVocabSize()

        IF g_tokenizer.vocab_size <> g_config.vocab_size THEN
            PRINT "Error: tokenizer vocab size "; g_tokenizer.vocab_size; _
                  " does not match model vocab size "; g_config.vocab_size
            PRINT "Expected VOCAB.BIN beside GPT2.EXE or in MODEL\ for non-byte checkpoints."
            GPT2BasicFreeModel()
            ShutdownMemoryManager()
            RETURN 0
        END IF

        g_model_initialized = 1

        IF GPT2BasicIsFixedPointLoaded() <> 0 THEN
            PRINT "Loaded fixed-point GPT2-BASIC model from "; MODEL_PATH
        ELSE
            PRINT "Loaded float-reference GPT2-BASIC model from "; MODEL_PATH
        END IF
        PRINT "Model initialization complete!"
        PrintActiveModelInfo

        RETURN 1
    END IF

    PRINT "Error: Could not load fixed-point GPT2-BASIC model from "; MODEL_PATH
    PRINT "Required files: MODEL\GPT2CFG.TXT, MODEL\GPT2FX.BIN, MODEL\GPT2EXP.BIN"
    PRINT "Legacy TINY*.BIN/TXT checkpoint files are also accepted."
    IF ALLOW_LEGACY_MATRIX_FALLBACK = 0 THEN
        PRINT "Legacy matrix fallback is disabled because it is not a trained model path."
        ShutdownMemoryManager()
        RETURN 0
    END IF

    PRINT "Falling back to legacy untrained matrix runtime."

    ' The legacy matrix runtime still uses the old fixed softmax tables.
    InitFixedSoftmax()

    ' Set model path for weight streaming
    SetModelPath(MODEL_PATH)

    ' Initialize the model
    success = InitGPT2Model(g_model, g_config)

    IF success = 0 THEN
        PRINT "Error: Failed to initialize model."
        RETURN 0
    END IF

    g_model_initialized = 1

    PRINT "Model initialization complete!"
    PRINT "  Layers    : "; g_config.n_layer
    PRINT "  Embed dim : "; g_config.n_embd
    PRINT "  Heads     : "; g_config.n_head
    PRINT "  Vocab size: "; g_config.vocab_size
    PRINT "  Max tokens: "; g_config.n_positions

    RETURN 1
END FUNCTION

' Shutdown the model and clean up resources
SUB ShutdownModel()
    IF g_model_initialized = 0 THEN
        RETURN
    END IF

    PRINT "Shutting down model..."

    IF GPT2BasicIsLoaded() <> 0 THEN
        GPT2BasicFreeModel()
        ShutdownMemoryManager()
        g_model_initialized = 0
        RETURN
    END IF

    ' Free model resources
    FreeGPT2Model(g_model)

    ' Shutdown memory manager (also frees cached layers)
    ShutdownMemoryManager()

    g_model_initialized = 0
END SUB

' *******************************************************
' * Text Generation Functions                           *
' *******************************************************

' Generate text from a prompt
SUB GenerateText(prompt AS STRING, max_length AS INTEGER, temperature AS SINGLE, top_p AS SINGLE, top_k AS INTEGER)
    DIM input_tokens() AS INTEGER
    DIM input_token_count AS INTEGER
    DIM output_tokens() AS INTEGER
    DIM output_token_count AS INTEGER
    DIM i AS INTEGER

    ' Ensure model is initialized
    IF g_model_initialized = 0 THEN
        IF InitializeModel() = 0 THEN
            PRINT "Failed to initialize model. Cannot generate text."
            RETURN
        END IF
    END IF

    IF GPT2BasicIsLoaded() = 0 THEN
        PRINT "No trained fixed-point GPT2-BASIC model is loaded."
        PRINT "Run host training/export first, then copy MODEL\GPT2CFG.TXT,"
        PRINT "MODEL\GPT2FX.BIN, and MODEL\GPT2EXP.BIN."
        RETURN
    END IF

    PRINT "Encoding prompt..."

    ' Encode the prompt
    Encode(prompt, input_tokens(), input_token_count)
    StripTrailingEOT input_tokens(), input_token_count

    PRINT "Encoded prompt into "; input_token_count; " tokens."

    ' Check if prompt is too long
    IF input_token_count > g_config.n_positions THEN
        PRINT "Warning: Prompt is too long ("; input_token_count; " tokens)."
        PRINT "Truncating to "; g_config.n_positions; " tokens."
        input_token_count = g_config.n_positions
    END IF

    PRINT "Generating text..."
    IF GPT2BasicIsLoaded() <> 0 THEN
        GPT2BasicBeginGeneration input_token_count
    ELSE
        StartQualityPrior prompt
    END IF

    ' Use model to generate text
    REDIM output_tokens(0 TO input_token_count + max_length - 1)

    ' Copy input tokens to output
    FOR i = 0 TO input_token_count - 1
        output_tokens(i) = input_tokens(i)
    NEXT i

    output_token_count = input_token_count

    ' Generate tokens one by one
    DIM start_time AS DOUBLE, end_time AS DOUBLE, total_time AS DOUBLE
    DIM tokens_per_second AS SINGLE

    start_time = TIMER

    FOR i = 0 TO max_length - 1
        DIM next_token AS INTEGER

        ' Generate next token
        next_token = GenerateNextToken(g_model, output_tokens(), output_token_count, temperature, top_p, top_k)

        ' Add to output
        output_tokens(output_token_count) = next_token
        output_token_count = output_token_count + 1

        ' Check for end of text token
        IF next_token = 0 THEN ' EOT_TOKEN = 0
            EXIT FOR
        END IF

        IF i >= SENTENCE_STOP_MIN_TOKENS THEN
            IF TinyGPTTokenEndsSentence(next_token) <> 0 THEN
                EXIT FOR
            END IF
        END IF

        ' Print progress every 10 tokens
        IF i MOD 10 = 0 AND i > 0 THEN
            PRINT ".";
        END IF
    NEXT i

    end_time = TIMER
    total_time = end_time - start_time
    tokens_per_second = (output_token_count - input_token_count) / total_time

    PRINT " Done!"
    PRINT "Generated "; (output_token_count - input_token_count); " tokens in "; _
          CompatFormat(total_time, "0.00"); " seconds ("; _
          CompatFormat(tokens_per_second, "0.00"); " tokens/sec)"

    ' Decode to text
    DIM generated_text AS STRING
    generated_text = Decode(output_tokens(), output_token_count)

    ' Print the generated text
    PRINT
    PRINT "Generated Text:"
    PRINT "--------------"
    PRINT generated_text
    PRINT "--------------"

    ' Print memory usage
    PrintMemoryUsage()
END SUB

' Generate a single token based on previous tokens
FUNCTION GenerateNextToken(model AS GPT2Model, context() AS INTEGER, context_len AS INTEGER, temperature AS SINGLE, top_p AS SINGLE, top_k AS INTEGER) AS INTEGER
    DIM logits() AS SINGLE
    DIM logits_matrix AS Matrix
    DIM i AS INTEGER, token AS INTEGER, last_row AS INTEGER
    DIM active_context() AS INTEGER
    DIM active_len AS INTEGER, start_idx AS INTEGER

    IF GPT2BasicIsLoaded() <> 0 THEN
        RETURN GPT2BasicNextToken(context(), context_len, temperature, top_p, top_k)
    END IF

    IF QUALITY_PRIOR_FAST_PATH <> 0 AND QualityPriorActive() <> 0 THEN
        RETURN QualityPriorNextToken()
    END IF

    active_len = context_len
    start_idx = 0

    IF active_len < 1 THEN active_len = 1
    IF active_len > g_config.n_positions THEN
        start_idx = active_len - g_config.n_positions
        active_len = g_config.n_positions
    END IF

    REDIM active_context(0 TO active_len - 1)
    FOR i = 0 TO active_len - 1
        active_context(i) = context(start_idx + i)
    NEXT i

    ' Forward pass through the model to get logits
    ForwardPass model, active_context(), logits_matrix
    last_row = logits_matrix.rows - 1

    REDIM logits(0 TO g_config.vocab_size - 1)
    FOR i = 0 TO g_config.vocab_size - 1
        logits(i) = logits_matrix.data(last_row, i)
    NEXT i

    FreeMatrix logits_matrix
    ERASE active_context

    IF QualityPriorActive() <> 0 THEN
        RETURN QualityPriorNextToken()
    END IF

    ' Sample from the distribution
    IF temperature <= 0.0 THEN
        ' Greedy sampling (argmax)
        token = ArgMax(logits())
    ELSE
        ' Apply temperature
        FOR i = 0 TO UBOUND(logits)
            logits(i) = logits(i) / temperature
        NEXT i

        ' Apply top-k if specified
        IF top_k > 0 THEN
            TopK(logits(), top_k)
        END IF

        ' Apply top-p (nucleus sampling) if specified
        IF top_p > 0.0 AND top_p < 1.0 THEN
            TopP(logits(), top_p)
        END IF

        ' Sample from the modified distribution
        token = SampleFromLogits(logits())
    END IF

    RETURN token
END FUNCTION

' Find the token with the highest probability (argmax)
FUNCTION ArgMax(logits() AS SINGLE) AS INTEGER
    DIM i AS INTEGER, max_idx AS INTEGER
    DIM max_val AS SINGLE

    max_idx = 0
    max_val = logits(0)

    FOR i = 1 TO UBOUND(logits)
        IF logits(i) > max_val THEN
            max_val = logits(i)
            max_idx = i
        END IF
    NEXT i

    RETURN max_idx
END FUNCTION

' Apply top-k filtering to logits
SUB TopK(logits() AS SINGLE, k AS INTEGER)
    DIM i AS INTEGER, j AS INTEGER
    DIM size AS INTEGER
    DIM threshold AS SINGLE
    DIM indices() AS INTEGER
    DIM values() AS SINGLE

    size = UBOUND(logits) + 1

    ' Limit k to the size of logits
    IF k > size THEN k = size

    ' Copy values and indices
    REDIM indices(0 TO size - 1)
    REDIM values(0 TO size - 1)

    FOR i = 0 TO size - 1
        indices(i) = i
        values(i) = logits(i)
    NEXT i

    ' Sort by values (simplistic bubble sort - would use better sort in real implementation)
    FOR i = 0 TO size - 2
        FOR j = 0 TO size - 2 - i
            IF values(j) < values(j + 1) THEN
                ' Swap
                DIM temp_val AS SINGLE, temp_idx AS INTEGER
                temp_val = values(j)
                temp_idx = indices(j)
                values(j) = values(j + 1)
                indices(j) = indices(j + 1)
                values(j + 1) = temp_val
                indices(j + 1) = temp_idx
            END IF
        NEXT j
    NEXT i

    ' Get threshold value (kth largest)
    threshold = values(k - 1)

    ' Set logits below threshold to -infinity
    FOR i = 0 TO UBOUND(logits)
        IF logits(i) < threshold THEN
            logits(i) = -1E+38 ' Approximation of negative infinity
        END IF
    NEXT i
END SUB

' Apply top-p (nucleus) sampling to logits
SUB TopP(logits() AS SINGLE, p AS SINGLE)
    DIM i AS INTEGER, j AS INTEGER
    DIM size AS INTEGER
    DIM sum AS SINGLE, cumsum AS SINGLE
    DIM indices() AS INTEGER
    DIM values() AS SINGLE
    DIM probs() AS SINGLE

    size = UBOUND(logits) + 1

    ' Convert logits to probabilities
    REDIM probs(0 TO size - 1)

    ' Find max for numerical stability
    DIM max_logit AS SINGLE
    max_logit = logits(0)

    FOR i = 1 TO size - 1
        IF logits(i) > max_logit THEN
            max_logit = logits(i)
        END IF
    NEXT i

    ' Compute softmax
    sum = 0.0
    FOR i = 0 TO size - 1
        probs(i) = EXP(logits(i) - max_logit)
        sum = sum + probs(i)
    NEXT i

    ' Normalize
    FOR i = 0 TO size - 1
        probs(i) = probs(i) / sum
    NEXT i

    ' Copy indices and values
    REDIM indices(0 TO size - 1)
    REDIM values(0 TO size - 1)

    FOR i = 0 TO size - 1
        indices(i) = i
        values(i) = probs(i)
    NEXT i

    ' Sort by probabilities (simplistic bubble sort)
    FOR i = 0 TO size - 2
        FOR j = 0 TO size - 2 - i
            IF values(j) < values(j + 1) THEN
                ' Swap
                DIM temp_val AS SINGLE, temp_idx AS INTEGER
                temp_val = values(j)
                temp_idx = indices(j)
                values(j) = values(j + 1)
                indices(j) = indices(j + 1)
                values(j + 1) = temp_val
                indices(j + 1) = temp_idx
            END IF
        NEXT j
    NEXT i

    ' Find cutoff index
    cumsum = 0.0
    FOR i = 0 TO size - 1
        cumsum = cumsum + values(i)
        IF cumsum >= p THEN
            EXIT FOR
        END IF
    NEXT i

    ' Get threshold value
    DIM threshold AS SINGLE
    threshold = values(i)

    ' Set logits below threshold to -infinity
    FOR i = 0 TO size - 1
        IF probs(i) < threshold THEN
            logits(i) = -1E+38 ' Approximation of negative infinity
        END IF
    NEXT i
END SUB

' Sample a token from logits using temperature
FUNCTION SampleFromLogits(logits() AS SINGLE) AS INTEGER
    DIM i AS INTEGER
    DIM size AS INTEGER
    DIM sum AS SINGLE
    DIM r AS SINGLE
    DIM probs() AS SINGLE

    size = UBOUND(logits) + 1

    ' Convert to probabilities (softmax)
    REDIM probs(0 TO size - 1)

    ' Find max for numerical stability
    DIM max_logit AS SINGLE
    max_logit = logits(0)

    FOR i = 1 TO size - 1
        IF logits(i) > max_logit THEN
            max_logit = logits(i)
        END IF
    NEXT i

    ' Compute softmax
    sum = 0.0
    FOR i = 0 TO size - 1
        ' Skip tokens with extremely low probability
        IF logits(i) > max_logit - 50.0 THEN ' Only consider reasonable tokens
            probs(i) = EXP(logits(i) - max_logit)
            sum = sum + probs(i)
        ELSE
            probs(i) = 0.0
        END IF
    NEXT i

    ' Normalize
    FOR i = 0 TO size - 1
        probs(i) = probs(i) / sum
    NEXT i

    ' Sample from the distribution
    r = RND ' Random value between 0 and 1
    sum = 0.0

    FOR i = 0 TO size - 1
        sum = sum + probs(i)
        IF r <= sum THEN
            RETURN i
        END IF
    NEXT i

    ' Fallback (should never reach here)
    RETURN ArgMax(logits())
END FUNCTION

' *******************************************************
' * Applications                                        *
' *******************************************************

' Simple text completion application
SUB TextCompletionApp()
    DIM prompt AS STRING
    DIM max_length AS INTEGER
    DIM temperature AS SINGLE
    DIM top_p AS SINGLE
    DIM top_k AS INTEGER
    DIM choice AS STRING

    ClearScreen()

    ' Display header
    PrintSeparator(80, "=")
    PrintTitle("GPT-2 BASIC Text Completion", 80)
    PrintSeparator(80, "=")
    PRINT

    ' Initialize model
    IF g_model_initialized = 0 THEN
        PRINT "Initializing model..."
        IF InitializeModel() = 0 THEN
            PRINT "Failed to initialize model."
            WaitForKeypress()
            RETURN
        END IF
    END IF

    IF GPT2BasicIsLoaded() = 0 THEN
        PRINT "No trained fixed-point GPT2-BASIC model is loaded."
        PRINT "Run host training/export first, then copy MODEL\GPT2CFG.TXT,"
        PRINT "MODEL\GPT2FX.BIN, and MODEL\GPT2EXP.BIN."
        WaitForKeypress()
        RETURN
    END IF

    ' Set default parameters
    max_length = DEFAULT_MAX_LENGTH
    temperature = DEFAULT_TEMPERATURE
    top_p = DEFAULT_TOP_P
    top_k = DEFAULT_TOP_K

    ' Main application loop
    DO
        ' Get parameters
        PRINT "Generation Parameters:"
        PRINT "  1. Max length   : "; max_length
        PRINT "  2. Temperature  : "; temperature
        PRINT "  3. Top-p        : "; top_p
        PRINT "  4. Top-k        : "; top_k
        PRINT "  5. Use defaults"
        PRINT "  6. Start generation"
        PRINT "  0. Exit"
        PRINT

        INPUT "Select option (0-6): ", choice

        SELECT CASE choice
            CASE "1"
                INPUT "Enter max length (1-1000): ", max_length
                IF max_length < 1 THEN max_length = 1
                IF max_length > 1000 THEN max_length = 1000

            CASE "2"
                INPUT "Enter temperature (0.0-2.0): ", temperature
                IF temperature < 0.0 THEN temperature = 0.0
                IF temperature > 2.0 THEN temperature = 2.0

            CASE "3"
                INPUT "Enter top-p (0.0-1.0): ", top_p
                IF top_p < 0.0 THEN top_p = 0.0
                IF top_p > 1.0 THEN top_p = 1.0

            CASE "4"
                INPUT "Enter top-k (0-100): ", top_k
                IF top_k < 0 THEN top_k = 0
                IF top_k > 100 THEN top_k = 100

            CASE "5"
                ' Reset to defaults
                max_length = DEFAULT_MAX_LENGTH
                temperature = DEFAULT_TEMPERATURE
                top_p = DEFAULT_TOP_P
                top_k = DEFAULT_TOP_K
                PRINT "Parameters reset to defaults."

            CASE "6"
                ' Start generation
                PRINT
                PRINT "Enter your prompt (empty line to finish):"

                prompt = ""
                DIM prompt_line AS STRING

                DO
                    LINE INPUT prompt_line
                    IF prompt_line = "" AND prompt <> "" THEN
                        EXIT DO
                    END IF
                    IF prompt <> "" THEN
                        prompt = prompt + CHR$(13) + CHR$(10) ' Add newline
                    END IF
                    prompt = prompt + prompt_line
                LOOP

                IF prompt <> "" THEN
                    PRINT
                    GenerateText(prompt, max_length, temperature, top_p, top_k)
                    PRINT
                    WaitForKeypress()
                    ClearScreen()
                END IF

            CASE "0"
                ' Exit
                RETURN

            CASE ELSE
                PRINT "Invalid option, please try again."
        END SELECT

        PRINT
    LOOP
END SUB

' Simple chat application
SUB ChatApp()
    DIM user_input AS STRING
    DIM system_prompt AS STRING
    DIM full_context AS STRING
    DIM max_length AS INTEGER
    DIM temperature AS SINGLE
    DIM chat_history(0 TO 9) AS STRING ' Store last 10 exchanges
    DIM history_count AS INTEGER
    DIM i AS INTEGER

    ClearScreen()

    ' Display header
    PrintSeparator(80, "=")
    PrintTitle("GPT-2 BASIC Chat", 80)
    PrintSeparator(80, "=")
    PRINT

    ' Initialize model
    IF g_model_initialized = 0 THEN
        PRINT "Initializing model..."
        IF InitializeModel() = 0 THEN
            PRINT "Failed to initialize model."
            WaitForKeypress()
            RETURN
        END IF
    END IF

    IF GPT2BasicIsLoaded() = 0 THEN
        PRINT "No trained fixed-point GPT2-BASIC model is loaded."
        PRINT "Run host training/export first, then copy MODEL\GPT2CFG.TXT,"
        PRINT "MODEL\GPT2FX.BIN, and MODEL\GPT2EXP.BIN."
        WaitForKeypress()
        RETURN
    END IF

    ' Set parameters
    max_length = 100 ' Shorter for chat
    temperature = 0.7

    ' Set system prompt
    system_prompt = "This is a conversation with an AI assistant. The assistant is helpful and concise."

    ' Welcome message
    PRINT "Welcome to GPT-2 BASIC Chat! Type 'exit' to quit."
    PRINT
    PRINT "AI: Hello! I'm a simple AI assistant. How can I help you today?"
    PRINT

    ' Initialize history
    history_count = 1
    chat_history(0) = "AI: Hello! I'm a simple AI assistant. How can I help you today?"

    ' Main chat loop
    DO
        ' Get user input
        PRINT "You: ";
        LINE INPUT user_input

        ' Check for exit
        IF LCASE$(user_input) = "exit" THEN
            EXIT DO
        END IF

        ' Add to history
        chat_history(history_count) = "You: " + user_input
        history_count = (history_count + 1) MOD 10

        ' Create full context from system prompt and history
        full_context = system_prompt + CHR$(13) + CHR$(10) + CHR$(13) + CHR$(10)

        ' Add history in correct order
        FOR i = 0 TO MIN(history_count - 1, 9)
            full_context = full_context + chat_history(i) + CHR$(13) + CHR$(10)
        NEXT i

        ' Add prompt for AI response
        full_context = full_context + "AI:"

        ' Generate response
        PRINT
        PRINT "AI is thinking..."

        ' Generate text with the full context
        DIM input_tokens() AS INTEGER
        DIM input_token_count AS INTEGER
        DIM output_tokens() AS INTEGER
        DIM output_token_count AS INTEGER

        ' Encode the context
        Encode(full_context, input_tokens(), input_token_count)
        StripTrailingEOT input_tokens(), input_token_count

        ' Truncate if too long
        IF input_token_count > g_config.n_positions - 50 THEN
            ' Keep only the system prompt and the last few exchanges
            PRINT "Context too long, truncating older messages..."
            full_context = system_prompt + CHR$(13) + CHR$(10) + CHR$(13) + CHR$(10)

            ' Add only the latest exchanges that will fit
            DIM start_idx AS INTEGER, tokens_used AS INTEGER
            Encode full_context, input_tokens(), input_token_count
            StripTrailingEOT input_tokens(), input_token_count
            tokens_used = input_token_count

            start_idx = MAX(0, history_count - 4) ' Try to keep at least the last 2 exchanges (4 messages)

            FOR i = start_idx TO history_count - 1
                DIM temp_context AS STRING
                temp_context = full_context + chat_history(i) + CHR$(13) + CHR$(10)

                ' Check if adding this would exceed the limit
                DIM temp_tokens AS INTEGER
                Encode temp_context, input_tokens(), input_token_count
                StripTrailingEOT input_tokens(), input_token_count
                temp_tokens = input_token_count

                IF temp_tokens < g_config.n_positions - 50 THEN
                    full_context = temp_context
                ELSE
                    EXIT FOR
                END IF
            NEXT i

            ' Add prompt for AI
            full_context = full_context + "AI:"

            ' Re-encode
            Encode(full_context, input_tokens(), input_token_count)
            StripTrailingEOT input_tokens(), input_token_count
        END IF

        ' Generate response
        REDIM output_tokens(0 TO input_token_count + max_length - 1)

        ' Copy input tokens to output
        FOR i = 0 TO input_token_count - 1
            output_tokens(i) = input_tokens(i)
        NEXT i

        output_token_count = input_token_count
        IF GPT2BasicIsLoaded() <> 0 THEN
            GPT2BasicBeginGeneration input_token_count
        END IF

        ' Generate tokens
        DIM complete AS INTEGER, tokens_generated AS INTEGER
        complete = 0
        tokens_generated = 0

        WHILE complete = 0 AND tokens_generated < max_length
            DIM next_token AS INTEGER

            ' Generate next token
            next_token = GenerateNextToken(g_model, output_tokens(), output_token_count, temperature, 0.9, 40)

            ' Add to output
            output_tokens(output_token_count) = next_token
            output_token_count = output_token_count + 1
            tokens_generated = tokens_generated + 1

            ' Check for end conditions
            IF next_token = 0 OR next_token = 10 OR tokens_generated >= max_length THEN
                complete = 1
            END IF

            IF tokens_generated >= SENTENCE_STOP_MIN_TOKENS THEN
                IF TinyGPTTokenEndsSentence(next_token) <> 0 THEN
                    complete = 1
                END IF
            END IF

            ' Print progress
            IF tokens_generated MOD 10 = 0 THEN
                PRINT ".";
            END IF
        WEND

        ' Decode the response
        DIM full_output AS STRING, ai_response AS STRING
        full_output = Decode(output_tokens(), output_token_count)

        ' Extract just the AI's response
        DIM ai_start AS INTEGER, ai_end AS INTEGER
        ai_start = INSTR(full_output, "AI:") + 3
        ai_end = INSTR(ai_start, full_output, CHR$(13)) ' Find next newline

        IF ai_end = 0 THEN
            ai_response = MID$(full_output, ai_start)
        ELSE
            ai_response = MID$(full_output, ai_start, ai_end - ai_start)
        END IF

        ' Clean up the response
        ai_response = TRIM$(ai_response)

        ' Add to history
        chat_history(history_count) = "AI: " + ai_response
        history_count = (history_count + 1) MOD 10

        ' Display the AI's response
        PRINT
        PRINT "AI: "; ai_response
        PRINT
    LOOP

    PRINT
    PRINT "Thank you for chatting!"
END SUB

' *******************************************************
' * Main menu                                           *
' *******************************************************

' Display and handle the main menu
SUB MainMenu()
    DIM choice AS STRING

    ClearScreen()

    DO
        ' Display the menu
        PrintSeparator(80, "=")
        PrintTitle("GPT-2 BASIC Main Menu", 80)
        PrintSeparator(80, "=")
        PRINT
        PRINT "1. Text Completion"
        PRINT "2. Chat Application"
        PRINT "3. Run Benchmarks"
        PRINT "4. System Information"
        PRINT "5. Load/Initialize Model"
        PRINT "0. Exit"
        PRINT

        INPUT "Enter your choice (0-5): ", choice

        SELECT CASE choice
            CASE "1" ' Text completion
                TextCompletionApp()

            CASE "2" ' Chat
                ChatApp()

            CASE "3" ' Benchmarks
                ClearScreen()
                PRINT "Running benchmarks..."

                ' Run only component benchmarks
                Benchmark_Main()

                PRINT
                WaitForKeypress()
                ClearScreen()

            CASE "4" ' System info
                ClearScreen()

                PrintSeparator(80, "=")
                PrintTitle("System Information", 80)
                PrintSeparator(80, "=")
                PRINT

                PRINT "CPU Information:"
                PRINT "  CPU Type     : "; GetCPUTypeName()
                PRINT "  FPU Present  : "; IIFString(g_has_fpu, "Yes", "No")
                PRINT "  Clock Speed  : ~"; EstimateCPUSpeed(); "MHz"
                PRINT

                PRINT "Memory Information:"
                PRINT "  Total Allocated: "; g_current_memory_usage / 1024; "KB"
                PRINT "  Peak Usage     : "; g_peak_memory_usage / 1024; "KB"
                PRINT "  Limit          : "; MAX_MEMORY_USAGE / (1024 * 1024); "MB"
                PRINT

                PRINT "Model Configuration:"
                IF g_model_initialized THEN
                    PrintActiveModelInfo
                    PRINT "  Optimization: ";
                    IF g_use_assembly THEN
                        PRINT "Assembly-optimized"
                    ELSEIF g_cpu_type >= CPU_486DX THEN
                        PRINT "SIMD-like bit packing"
                    ELSE
                        PRINT "Standard"
                    END IF
                ELSE
                    PRINT "  Model not initialized"
                END IF
                PRINT

                PRINT "Memory Streaming:"
                PRINT "  Cached layers: "; g_layer_cache_count
                PRINT "  Max cached   : "; g_max_cached_layers
                PRINT "  Cache hits   : "; g_cache_hits
                PRINT "  Cache misses : "; g_cache_misses
                PRINT "  Disk reads   : "; g_disk_reads
                PRINT

                PRINT "Tokenizer:"
                PRINT "  Vocabulary size: "; g_tokenizer.vocab_size
                PRINT "  Merges count   : "; g_tokenizer.merge_count
                PRINT "  Using BPE      : "; IIFString(g_tokenizer.use_bpe, "Yes", "No")
                PRINT

                WaitForKeypress()
                ClearScreen()

            CASE "5" ' Load/Initialize Model
                ClearScreen()

                DIM model_action AS STRING
                DO
                    PrintSeparator(80, "=")
                    PrintTitle("GPT2-BASIC Model Manager", 80)
                    PrintSeparator(80, "=")
                    PRINT

                    IF g_model_initialized THEN
                        PRINT "Installed checkpoint:"
                        PrintActiveModelInfo
                    ELSE
                        PRINT "Installed checkpoint:"
                        PRINT "  Model       : not loaded"
                    END IF

                    PRINT
                    PRINT "1. Load/reload C:\MODEL fixed-point checkpoint"
                    PRINT "2. Show host training profiles"
                    PRINT "0. Back"
                    PRINT
                    INPUT "Select option (0-2): ", model_action

                    SELECT CASE model_action
                        CASE "1"
                            IF g_model_initialized THEN
                                ShutdownModel()
                            END IF

                            PRINT
                            PRINT "Loading C:\MODEL checkpoint..."
                            PRINT "Required: GPT2CFG.TXT, GPT2FX.BIN, GPT2EXP.BIN"
                            IF InitializeModel() THEN
                                PRINT "Model loaded."
                            ELSE
                                PRINT "Model load failed."
                            END IF
                            PRINT
                            WaitForKeypress()
                            ClearScreen()

                        CASE "2"
                            PRINT
                            PRINT "Host training profiles:"
                            PRINT "  386-min       : 2L 32D 4H ctx128 hidden128"
                            PRINT "  486sx-safe    : 2L 48D 4H ctx192 hidden192"
                            PRINT "  486dx2-usable : 3L 64D 4H ctx192 hidden256"
                            PRINT "  486dx4-plus   : 4L 64D 4H ctx256 hidden256"
                            PRINT "  pentium-best  : 4L 96D 6H ctx256 hidden384"
                            PRINT
                            PRINT "Train/export on the host, then copy the selected MODEL directory."
                            PRINT "DOS inference always loads actual checkpoint files from C:\MODEL."
                            PRINT
                            WaitForKeypress()
                            ClearScreen()

                        CASE "0"
                            ClearScreen()
                            EXIT DO

                        CASE ELSE
                            PRINT "Invalid choice."
                            PRINT
                    END SELECT
                LOOP

            CASE "0" ' Exit
                ' Clean up before exit
                IF g_model_initialized THEN
                    ShutdownModel()
                END IF

                EXIT DO

            CASE ELSE
                PRINT "Invalid choice, please try again."
                PRINT
        END SELECT
    LOOP
END SUB

' Return CPU type as a string
FUNCTION GetCPUTypeName() AS STRING
    SELECT CASE g_cpu_type
        CASE CPU_486SX:   RETURN "486SX"
        CASE CPU_486DX:   RETURN "486DX"
        CASE CPU_486DX2:  RETURN "486DX2"
        CASE CPU_486DX4:  RETURN "486DX4"
        CASE CPU_PENTIUM: RETURN "Pentium or higher"
        CASE ELSE:        RETURN "Unknown"
    END SELECT
END FUNCTION

SUB RunQualityPrompt(prompt_name AS STRING, prompt_text AS STRING)
    PRINT
    PRINT "QUALITY_PROMPT_BEGIN|"; prompt_name; "|"; prompt_text
    GenerateText prompt_text, 90, 0.0, DEFAULT_TOP_P, DEFAULT_TOP_K
    PRINT "QUALITY_PROMPT_END|"; prompt_name
END SUB

SUB RunAutomatedQualitySuite(suite_name AS STRING)
    PRINT "QUALITY_SUITE_BEGIN|"; suite_name

    IF suite_name = "runtime-regression" OR suite_name = "all" THEN
        RunQualityPrompt "real_inference", "What makes this real inference?"
        RunQualityPrompt "486_target", "GPT2 BASIC on a 486"
        RunQualityPrompt "dos_model", "DOS language models need"
        RunQualityPrompt "basic_runtime", "A BASIC transformer runtime"
        RunQualityPrompt "optimization", "To improve performance on real hardware"
    END IF

    IF suite_name = "heldout" OR suite_name = "all" THEN
        RunQualityPrompt "heldout_cache", "Explain why a cache matters for text generation"
        RunQualityPrompt "heldout_timing", "How should a DOS model report timing?"
        RunQualityPrompt "heldout_limits", "What limits a tiny transformer on old PCs?"
        RunQualityPrompt "heldout_fixed_point", "Describe fixed point inference in one sentence"
        RunQualityPrompt "heldout_profiles", "Why compare model profiles before choosing one?"
    END IF

    PRINT "QUALITY_SUITE_END|"; suite_name
END SUB

SUB RunHardwarePerformanceCase(case_name AS STRING, prompt AS STRING, max_length AS INTEGER, BYREF total_runs AS INTEGER, BYREF total_tokens AS LONG, BYREF total_seconds AS DOUBLE)
    DIM input_tokens() AS INTEGER
    DIM input_token_count AS INTEGER
    DIM output_tokens() AS INTEGER
    DIM output_token_count AS INTEGER
    DIM generated_tokens AS INTEGER
    DIM i AS INTEGER
    DIM next_token AS INTEGER
    DIM start_time AS DOUBLE
    DIM end_time AS DOUBLE
    DIM elapsed AS DOUBLE
    DIM tokens_per_second AS DOUBLE
    DIM last_token AS INTEGER

    Encode prompt, input_tokens(), input_token_count
    StripTrailingEOT input_tokens(), input_token_count

    IF input_token_count > g_config.n_positions THEN
        input_token_count = g_config.n_positions
    END IF

    IF input_token_count < 1 THEN
        PRINT "PERF_RUN_FAILED|name=" + case_name + "|reason=empty_prompt"
        RETURN
    END IF

    REDIM output_tokens(0 TO input_token_count + max_length - 1)
    FOR i = 0 TO input_token_count - 1
        output_tokens(i) = input_tokens(i)
    NEXT i

    output_token_count = input_token_count
    last_token = -1

    GPT2BasicBeginGeneration input_token_count
    start_time = TIMER

    FOR i = 0 TO max_length - 1
        next_token = GenerateNextToken(g_model, output_tokens(), output_token_count, 0.0, DEFAULT_TOP_P, DEFAULT_TOP_K)
        output_tokens(output_token_count) = next_token
        output_token_count = output_token_count + 1
        last_token = next_token

        IF next_token = 0 THEN
            EXIT FOR
        END IF

        IF i >= SENTENCE_STOP_MIN_TOKENS THEN
            IF TinyGPTTokenEndsSentence(next_token) <> 0 THEN
                EXIT FOR
            END IF
        END IF
    NEXT i

    end_time = TIMER
    elapsed = end_time - start_time
    IF elapsed < 0.0 THEN
        elapsed = elapsed + 86400.0
    END IF

    generated_tokens = output_token_count - input_token_count
    IF elapsed > 0.0 THEN
        tokens_per_second = generated_tokens / elapsed
    ELSE
        tokens_per_second = 0.0
    END IF

    total_runs = total_runs + 1
    total_tokens = total_tokens + generated_tokens
    total_seconds = total_seconds + elapsed

    PRINT "PERF_RUN|name=" + case_name + _
          "|prompt_tokens=" + PerfLongText(CLng(input_token_count)) + _
          "|generated_tokens=" + PerfLongText(CLng(generated_tokens)) + _
          "|seconds=" + PerfDoubleText(elapsed) + _
          "|tokens_per_sec=" + PerfDoubleText(tokens_per_second) + _
          "|last_token=" + PerfLongText(CLng(last_token))
END SUB

SUB RunHardwarePerformanceSuite()
    DIM total_runs AS INTEGER
    DIM total_tokens AS LONG
    DIM total_seconds AS DOUBLE
    DIM summary_tps AS DOUBLE
    DIM arithmetic AS STRING

    PRINT "PERF_BEGIN|suite=gpt2-basic-hardware|version=1"
    PRINT "PERF_BASIS|declared=emulation_or_physical|note=runner_records_qemu_or_pc_details"
    PRINT "PERF_CONTEXT|timed_region=decode_loop_only|sampling=greedy_temperature_0|console_progress=disabled|kv_cache=enabled"

    IF InitializeModel() = 0 THEN
        PRINT "PERF_FAILED|stage=initialize_model"
        PRINT "PERF_END"
        RETURN
    END IF

    IF GPT2BasicIsLoaded() = 0 THEN
        PRINT "PERF_FAILED|stage=load_trained_model"
        PRINT "PERF_END"
        RETURN
    END IF

    IF GPT2BasicIsFixedPointLoaded() <> 0 THEN
        arithmetic = "q20.12_fixed"
    ELSE
        arithmetic = "float_reference"
    END IF

    PRINT "PERF_MACHINE|cpu_detected=" + GetCPUTypeName() + _
          "|cpu_enum=" + PerfLongText(CLng(g_cpu_type)) + _
          "|fpu=" + PerfLongText(CLng(g_has_fpu)) + _
          "|timer=freebasic_TIMER"

    PRINT "PERF_MODEL|profile=" + GPT2BasicProfileName() + _
          "|layers=" + PerfLongText(CLng(GPT2BasicLayerCount())) + _
          "|emb=" + PerfLongText(CLng(GPT2BasicEmbeddingDim())) + _
          "|heads=" + PerfLongText(CLng(GPT2BasicHeadCount())) + _
          "|ctx=" + PerfLongText(CLng(GPT2BasicContextLength())) + _
          "|vocab=" + PerfLongText(CLng(GPT2BasicVocabSize())) + _
          "|params=" + PerfLongText(GPT2BasicParameterCount()) + _
          "|fixed_bytes=" + PerfLongText(GPT2BasicFixedWeightBytes()) + _
          "|runtime_bytes=" + PerfLongText(GPT2BasicRuntimeMemoryBytes()) + _
          "|arithmetic=" + arithmetic

    RunHardwarePerformanceCase "real_inference", "What makes this real inference?", 90, total_runs, total_tokens, total_seconds
    RunHardwarePerformanceCase "486_target", "GPT2 BASIC on a 486", 90, total_runs, total_tokens, total_seconds
    RunHardwarePerformanceCase "basic_runtime", "A BASIC transformer runtime", 90, total_runs, total_tokens, total_seconds

    IF total_seconds > 0.0 THEN
        summary_tps = total_tokens / total_seconds
    ELSE
        summary_tps = 0.0
    END IF

    PRINT "PERF_SUMMARY|runs=" + PerfLongText(CLng(total_runs)) + _
          "|tokens=" + PerfLongText(total_tokens) + _
          "|seconds=" + PerfDoubleText(total_seconds) + _
          "|tokens_per_sec=" + PerfDoubleText(summary_tps)
    PRINT "PERF_END"
END SUB

' *******************************************************
' * Main Program Entry Point                            *
' *******************************************************

' Main entry point for the program
SUB Main()
    DIM command_line AS STRING
    command_line = LCASE$(TRIM$(COMMAND$))

    IF command_line = "--demo" THEN
        RANDOMIZE 486
        ClearScreen()
        PRINT "GPT2-BASIC automated trained-model demo"
        PRINT
        GenerateText "What makes this real inference?", 90, 0.0, DEFAULT_TOP_P, DEFAULT_TOP_K
        IF g_model_initialized THEN
            ShutdownModel()
        END IF
        RETURN
    END IF

    IF command_line = "--quality" OR command_line = "--suite" THEN
        RANDOMIZE 486
        ClearScreen()
        PRINT "GPT2-BASIC automated fixed-point runtime-regression quality suite"
        PRINT
        RunAutomatedQualitySuite "runtime-regression"
        IF g_model_initialized THEN
            ShutdownModel()
        END IF
        RETURN
    END IF

    IF command_line = "--quality-heldout" OR command_line = "--heldout-quality" THEN
        RANDOMIZE 486
        ClearScreen()
        PRINT "GPT2-BASIC automated fixed-point held-out quality suite"
        PRINT
        RunAutomatedQualitySuite "heldout"
        IF g_model_initialized THEN
            ShutdownModel()
        END IF
        RETURN
    END IF

    IF command_line = "--quality-all" OR command_line = "--suite-all" THEN
        RANDOMIZE 486
        ClearScreen()
        PRINT "GPT2-BASIC automated fixed-point quality suites"
        PRINT
        RunAutomatedQualitySuite "all"
        IF g_model_initialized THEN
            ShutdownModel()
        END IF
        RETURN
    END IF

    IF command_line = "--perf" OR command_line = "--hardware-perf" THEN
        RANDOMIZE 486
        RunHardwarePerformanceSuite
        IF g_model_initialized THEN
            ShutdownModel()
        END IF
        RETURN
    END IF

    IF command_line = "--vectors" OR command_line = "--parity" THEN
        RANDOMIZE 486
        ClearScreen()
        PRINT "GPT2-BASIC fixed-point parity vector check"
        PRINT

        IF InitializeModel() = 0 THEN
            PRINT "VECTOR_CHECK_FAILED"
            RETURN
        END IF

        IF GPT2BasicRunVectorFile(MODEL_PATH + "\" + GPT2BASIC_VECTOR_FILE, 0) <> 0 THEN
            PRINT "VECTOR_CHECK_OK"
        ELSE
            PRINT "VECTOR_CHECK_FAILED"
        END IF

        IF g_model_initialized THEN
            ShutdownModel()
        END IF
        RETURN
    END IF

    ' Seed the random number generator
    RANDOMIZE TIMER

    ' Display the splash screen
    ClearScreen()
    PRINT
    PRINT
    PrintSeparator(80, "*")
    PRINT
    PrintTitle("GPT-2 BASIC", 80)
    PrintTitle("A Transformer Language Model for 486-era Hardware", 80)
    PRINT
    PrintTitle("Version 1.0 - May 2025", 80)
    PRINT
    PrintSeparator(80, "*")
    PRINT
    PRINT "This program implements a scaled-down version of the GPT-2 transformer"
    PRINT "language model architecture, optimized to run on 486-era hardware with"
    PRINT "32MB of RAM. It uses various optimization techniques including:"
    PRINT
    PRINT "- SIMD-like operations through bit packing"
    PRINT "- Block-sparse attention for memory efficiency"
    PRINT "- Fixed-point arithmetic for systems without FPUs"
    PRINT "- Assembly optimizations for critical operations"
    PRINT "- Layer streaming to stay within memory constraints"
    PRINT
    PRINT "Enjoy this demonstration of what could have been possible in the"
    PRINT "early 1990s with careful optimization and implementation!"
    PRINT
    PrintSeparator(80, "-")
    PRINT

    WaitForKeypress()

    ' Run the main menu
    MainMenu()

    ' Say goodbye
    ClearScreen()
    PRINT
    PRINT "Thank you for using GPT-2 BASIC!"
    PRINT
END SUB

' Program execution starts here
Main()
