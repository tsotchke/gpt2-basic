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
#INCLUDE "src/transformer_components.bas"
#INCLUDE "src/model.bas"
#INCLUDE "src/file_io.bas"
#INCLUDE "src/benchmark.bas"

' *******************************************************
' * Constants and Configuration                         *
' *******************************************************

' Default model configuration settings
CONST MODEL_PATH = "models/gpt2-basic"
CONST VOCAB_PATH = "models/vocab.bin"
CONST CONFIG_PATH = "models/config.txt"

' Default generation parameters
CONST DEFAULT_MAX_LENGTH = 50
CONST DEFAULT_TEMPERATURE = 0.8
CONST DEFAULT_TOP_P = 0.9
CONST DEFAULT_TOP_K = 40

' *******************************************************
' * Global Variables                                    *
' *******************************************************

DIM SHARED g_model AS GPT2Model
DIM SHARED g_config AS ModelConfig
DIM SHARED g_model_initialized AS INTEGER

' *******************************************************
' * Utility Functions                                   *
' *******************************************************

' Clear the screen - DOS specific
SUB ClearScreen()
    SYSTEM("CLS")
END SUB

' Wait for a key press
SUB WaitForKeypress()
    PRINT "Press any key to continue..."
    WHILE INKEY$ = "": WEND
END SUB

' Print a centered title
SUB PrintTitle(title AS STRING, width AS INTEGER)
    DIM pad_length AS INTEGER
    pad_length = (width - LEN(title)) \ 2
    PRINT STRING$(pad_length, " "); title
END SUB

' Print a separator line
SUB PrintSeparator(width AS INTEGER, char AS STRING)
    PRINT STRING$(width, char)
END SUB

' Print memory usage information
SUB PrintMemoryUsage()
    PRINT "Memory Usage:"
    PRINT "  Current: "; g_current_memory_usage / (1024 * 1024); "MB"
    PRINT "  Peak   : "; g_peak_memory_usage / (1024 * 1024); "MB"
    PRINT "  Available: "; GetAvailableMemory() / (1024 * 1024); "MB"
END SUB

' *******************************************************
' * Model Setup and Initialization                      *
' *******************************************************

' Initialize the model and all components
FUNCTION InitializeModel() AS INTEGER
    DIM success AS INTEGER
    
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
    
    ' Initialize fixed-point softmax
    InitFixedSoftmax()
    
    ' Initialize tokenizer
    InitializeDefaultTokenizer()
    
    ' Attempt to load vocabulary
    success = 0
    ON ERROR GOTO vocab_error
    LoadDefaultVocabulary(VOCAB_PATH)
    success = 1
    
vocab_error:
    ON ERROR GOTO 0
    
    IF NOT success THEN
        PRINT "Warning: Could not load vocabulary from "; VOCAB_PATH
        PRINT "Using default byte-level vocabulary."
    END IF
    
    ' Load model configuration
    success = LoadModelConfig(g_config, CONFIG_PATH)
    
    IF NOT success THEN
        PRINT "Warning: Could not load configuration from "; CONFIG_PATH
        PRINT "Using default configuration."
        
        ' Set default configuration
        InitModelConfig(g_config)
    END IF
    
    ' Set model path for weight streaming
    SetModelPath(MODEL_PATH)
    
    ' Initialize the model
    success = InitGPT2Model(g_model, g_config)
    
    IF NOT success THEN
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
    IF NOT g_model_initialized THEN
        RETURN
    END IF
    
    PRINT "Shutting down model..."
    
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
    IF NOT g_model_initialized THEN
        IF NOT InitializeModel() THEN
            PRINT "Failed to initialize model. Cannot generate text."
            RETURN
        END IF
    END IF
    
    PRINT "Encoding prompt..."
    
    ' Encode the prompt
    Encode(prompt, input_tokens(), input_token_count)
    
    PRINT "Encoded prompt into "; input_token_count; " tokens."
    
    ' Check if prompt is too long
    IF input_token_count > g_config.n_positions THEN
        PRINT "Warning: Prompt is too long ("; input_token_count; " tokens)."
        PRINT "Truncating to "; g_config.n_positions; " tokens."
        input_token_count = g_config.n_positions
    END IF
    
    PRINT "Generating text..."
    
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
          FORMAT(total_time, "0.00"); " seconds ("; _
          FORMAT(tokens_per_second, "0.00"); " tokens/sec)"
    
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
    DIM i AS INTEGER, token AS INTEGER
    
    ' Forward pass through the model to get logits
    ForwardPass(model, context(), context_len, logits())
    
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
SUB TopK(BYREF logits() AS SINGLE, k AS INTEGER)
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
SUB TopP(BYREF logits() AS SINGLE, p AS SINGLE)
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
    IF NOT g_model_initialized THEN
        PRINT "Initializing model..."
        IF NOT InitializeModel() THEN
            PRINT "Failed to initialize model."
            WaitForKeypress()
            RETURN
        END IF
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
                DIM line AS STRING
                
                DO
                    LINE INPUT line
                    IF line = "" AND prompt <> "" THEN
                        EXIT DO
                    END IF
                    IF prompt <> "" THEN
                        prompt = prompt + CHR$(13) + CHR$(10) ' Add newline
                    END IF
                    prompt = prompt + line
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
    
    ClearScreen()
    
    ' Display header
    PrintSeparator(80, "=")
    PrintTitle("GPT-2 BASIC Chat", 80)
    PrintSeparator(80, "=")
    PRINT
    
    ' Initialize model
    IF NOT g_model_initialized THEN
        PRINT "Initializing model..."
        IF NOT InitializeModel() THEN
            PRINT "Failed to initialize model."
            WaitForKeypress()
            RETURN
        END IF
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
        
        ' Truncate if too long
        IF input_token_count > g_config.n_positions - 50 THEN
            ' Keep only the system prompt and the last few exchanges
            PRINT "Context too long, truncating older messages..."
            full_context = system_prompt + CHR$(13) + CHR$(10) + CHR$(13) + CHR$(10)
            
            ' Add only the latest exchanges that will fit
            DIM start_idx AS INTEGER, tokens_used AS INTEGER
            tokens_used = Encode(full_context, input_tokens(), input_token_count)
            
            start_idx = MAX(0, history_count - 4) ' Try to keep at least the last 2 exchanges (4 messages)
            
            FOR i = start_idx TO history_count - 1
                DIM temp_context AS STRING
                temp_context = full_context + chat_history(i) + CHR$(13) + CHR$(10)
                
                ' Check if adding this would exceed the limit
                DIM temp_tokens AS INTEGER
                temp_tokens = Encode(temp_context, input_tokens(), input_token_count)
                
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
        END IF
        
        ' Generate response
        REDIM output_tokens(0 TO input_token_count + max_length - 1)
        
        ' Copy input tokens to output
        FOR i = 0 TO input_token_count - 1
            output_tokens(i) = input_tokens(i)
        NEXT i
        
        output_token_count = input_token_count
        
        ' Generate tokens
        DIM complete AS INTEGER, tokens_generated AS INTEGER
        complete = 0
        tokens_generated = 0
        
        WHILE NOT complete AND tokens_generated < max_length
            DIM next_token AS INTEGER
            
            ' Generate next token
            next_token = GenerateNextToken(g_model, output_tokens(), output_token_count, temperature, 0.9, 40)
            
            ' Add to output
            output_tokens(output_token_count) = next_token
            output_token_count = output_token_count + 1
            tokens_generated = tokens_generated + 1
            
            ' Check for end conditions
            IF next_token = 0 OR ' EOT token
               next_token = 10 OR ' Newline
               tokens_generated >= max_length THEN
                complete = 1
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
                PRINT "  FPU Present  : "; IIF(g_has_fpu, "Yes", "No")
                PRINT "  Clock Speed  : ~"; EstimateCPUSpeed(); "MHz"
                PRINT
                
                PRINT "Memory Information:"
                PRINT "  Total Allocated: "; g_current_memory_usage / 1024; "KB"
                PRINT "  Peak Usage     : "; g_peak_memory_usage / 1024; "KB"
                PRINT "  Limit          : "; MAX_MEMORY_USAGE / (1024 * 1024); "MB"
                PRINT
                
                PRINT "Model Configuration:"
                IF g_model_initialized THEN
                    PRINT "  Layers      : "; g_config.n_layer
                    PRINT "  Embed dim   : "; g_config.n_embd
                    PRINT "  Heads       : "; g_config.n_head
                    PRINT "  Vocab size  : "; g_config.vocab_size
                    PRINT "  Max tokens  : "; g_config.n_positions
                    PRINT "  Optimization: "; 
                    IF g_use_assembly THEN
                        PRINT "Assembly-optimized"
                    ELSEIF g_use_simd THEN
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
                PRINT "  Using BPE      : "; IIF(g_tokenizer.use_bpe, "Yes", "No")
                PRINT
                
                WaitForKeypress()
                ClearScreen()
                
            CASE "5" ' Load/Initialize Model
                ClearScreen()
                
                PrintSeparator(80, "=")
                PrintTitle("Model Initialization", 80)
                PrintSeparator(80, "=")
                PRINT
                
                IF g_model_initialized THEN
                    PRINT "Model is already initialized."
                    PRINT "Do you want to reinitialize the model? (y/n)"
                    
                    DIM reinit AS STRING
                    INPUT reinit
                    
                    IF LCASE$(reinit) <> "y" THEN
                        PRINT "Keeping current model."
                        WaitForKeypress()
                        ClearScreen()
                        CONTINUE DO
                    END IF
                    
                    ' Shutdown current model
                    ShutdownModel()
                END IF
                
                ' Show model options
                PRINT "Select model configuration:"
                PRINT "  1. Tiny (4 layers, 128 embedding)"
                PRINT "  2. Small (6 layers, 256 embedding)"
                PRINT "  3. Medium (8 layers, 512 embedding)"
                PRINT "  4. Load from config file"
                PRINT
                
                DIM model_choice AS STRING
                INPUT "Enter choice (1-4): ", model_choice
                
                ' Set configuration based on choice
                SELECT CASE model_choice
                    CASE "1" ' Tiny
                        InitModelConfig(g_config)
                        g_config.n_layer = 4
                        g_config.n_embd = 128
                        g_config.n_head = 4
                        g_config.vocab_size = 16384
                        g_config.n_positions = 512
                        
                    CASE "2" ' Small
                        InitModelConfig(g_config)
                        g_config.n_layer = 6
                        g_config.n_embd = 256
                        g_config.n_head = 8
                        g_config.vocab_size = 16384
                        g_config.n_positions = 512
                        
                    CASE "3" ' Medium
                        InitModelConfig(g_config)
                        g_config.n_layer = 8
                        g_config.n_embd = 512
                        g_config.n_head = 16
                        g_config.vocab_size = 32768
                        g_config.n_positions = 1024
                        
                    CASE "4" ' Load from config
                        DIM config_path AS STRING
                        PRINT "Enter path to config file (empty for default: "; CONFIG_PATH; "):"
                        LINE INPUT config_path
                        
                        IF config_path = "" THEN
                            config_path = CONFIG_PATH
                        END IF
                        
                        IF NOT LoadModelConfig(g_config, config_path) THEN
                            PRINT "Error loading config from "; config_path
                            PRINT "Using default configuration."
                            InitModelConfig(g_config)
                        END IF
                        
                    CASE ELSE
                        PRINT "Invalid choice, using default configuration."
                        InitModelConfig(g_config)
                END SELECT
                
                ' Initialize the model
                PRINT
                PRINT "Initializing model..."
                
                IF InitializeModel() THEN
                    PRINT "Model initialized successfully!"
                ELSE
                    PRINT "Failed to initialize model."
                END IF
                
                PRINT
                WaitForKeypress()
                ClearScreen()
                
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

' *******************************************************
' * Main Program Entry Point                            *
' *******************************************************

' Main entry point for the program
SUB Main()
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
