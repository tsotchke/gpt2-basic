' *******************************************************
' * GPT-2 BASIC Production Entry Point                 *
' *******************************************************
' * This program builds the release GPT2.EXE surface.  *
' * It intentionally excludes the legacy matrix, sparse *
' * attention, benchmark, diagnostic prior, and lab UI  *
' * modules. Production generation must load MODEL\ and *
' * run the real tokenizer plus fixed-point runtime.    *
' *******************************************************

#INCLUDE "src/tokenizer.bas"

DIM SHARED g_prod_current_memory AS LONG
DIM SHARED g_prod_peak_memory AS LONG

DECLARE SUB TrackAllocation(size AS LONG)
DECLARE SUB TrackDeallocation(size AS LONG)

SUB TrackAllocation(size AS LONG)
    IF size <= 0 THEN RETURN
    g_prod_current_memory = g_prod_current_memory + size
    IF g_prod_current_memory > g_prod_peak_memory THEN g_prod_peak_memory = g_prod_current_memory
END SUB

SUB TrackDeallocation(size AS LONG)
    IF size <= 0 THEN RETURN
    g_prod_current_memory = g_prod_current_memory - size
    IF g_prod_current_memory < 0 THEN g_prod_current_memory = 0
END SUB

#INCLUDE "src/real_gpt.bas"

CONST MODEL_PATH = "MODEL"
CONST VOCAB_PATH = "VOCAB.BIN"
CONST MODEL_VOCAB_PATH = "MODEL\VOCAB.BIN"
CONST DEFAULT_MAX_LENGTH = 220
CONST DEFAULT_TOP_P = 0.9
CONST DEFAULT_TOP_K = 40
CONST SENTENCE_STOP_MIN_TOKENS = 30

DIM SHARED g_model_initialized AS INTEGER
DIM SHARED g_prod_kernel_perf_request AS INTEGER

DECLARE SUB ClearScreen()
DECLARE SUB WaitForKeypress()
DECLARE SUB StripTrailingEOT(tokens() AS INTEGER, BYREF token_count AS INTEGER)
DECLARE FUNCTION ProdLongText(value AS LONG) AS STRING
DECLARE FUNCTION ProdDoubleText(value AS DOUBLE) AS STRING
DECLARE FUNCTION TraceSafeText(value AS STRING) AS STRING
DECLARE FUNCTION TraceTokenizerModeName(mode_value AS INTEGER) AS STRING
DECLARE FUNCTION InitializeModel() AS INTEGER
DECLARE SUB ShutdownModel()
DECLARE SUB PrintActiveModelInfo()
DECLARE FUNCTION GenerateNextToken(context() AS INTEGER, context_len AS INTEGER, temperature AS SINGLE, top_p AS SINGLE, top_k AS INTEGER) AS INTEGER
DECLARE SUB GenerateText(prompt AS STRING, max_length AS INTEGER, temperature AS SINGLE, top_p AS SINGLE, top_k AS INTEGER)
DECLARE SUB RunTraceMode(prompt AS STRING, max_length AS INTEGER)
DECLARE SUB RunQualityPrompt(prompt_name AS STRING, prompt_text AS STRING)
DECLARE SUB RunAutomatedQualitySuite(suite_name AS STRING)
DECLARE SUB RunHardwarePerformanceCase(case_name AS STRING, prompt AS STRING, max_length AS INTEGER, BYREF total_runs AS INTEGER, BYREF total_tokens AS LONG, BYREF total_seconds AS DOUBLE)
DECLARE SUB RunHardwarePerformanceSuite()
DECLARE SUB InteractiveCompletion()
DECLARE SUB Main()

SUB ClearScreen()
    CLS
END SUB

SUB WaitForKeypress()
    PRINT "Press any key to continue..."
    WHILE INKEY$ = "": WEND
END SUB

SUB StripTrailingEOT(tokens() AS INTEGER, BYREF token_count AS INTEGER)
    IF token_count <= 0 THEN RETURN
    IF tokens(token_count - 1) = 0 THEN token_count = token_count - 1
END SUB

FUNCTION ProdLongText(value AS LONG) AS STRING
    RETURN LTRIM$(STR$(value))
END FUNCTION

FUNCTION ProdDoubleText(value AS DOUBLE) AS STRING
    RETURN LTRIM$(STR$(value))
END FUNCTION

FUNCTION TraceSafeText(value AS STRING) AS STRING
    DIM result AS STRING
    DIM i AS INTEGER
    DIM ch AS STRING
    DIM code AS INTEGER

    result = ""
    FOR i = 1 TO LEN(value)
        ch = MID$(value, i, 1)
        code = ASC(ch)
        IF ch = "|" THEN
            result = result + "/"
        ELSEIF code = 13 OR code = 10 OR code = 9 THEN
            result = result + " "
        ELSEIF code < 32 OR code > 126 THEN
            result = result + "."
        ELSE
            result = result + ch
        END IF
    NEXT i

    RETURN result
END FUNCTION

FUNCTION TraceTokenizerModeName(mode_value AS INTEGER) AS STRING
    IF mode_value = TOKENIZER_MODE_BYTE THEN RETURN "byte"
    IF mode_value = TOKENIZER_MODE_BPE THEN RETURN "bpe"
    IF mode_value = TOKENIZER_MODE_LEXICON THEN RETURN "lexicon"
    RETURN "unknown"
END FUNCTION

FUNCTION InitializeModel() AS INTEGER
    DIM success AS INTEGER
    DIM vocab_source AS STRING

    IF g_model_initialized <> 0 THEN RETURN 1

    PRINT "Initializing GPT2-BASIC production runtime..."

    InitializeDefaultTokenizer
    vocab_source = ""
    IF DIR(VOCAB_PATH) <> "" THEN
        vocab_source = VOCAB_PATH
    ELSEIF DIR(MODEL_VOCAB_PATH) <> "" THEN
        vocab_source = MODEL_VOCAB_PATH
    END IF

    IF vocab_source <> "" THEN
        success = 0
        ON ERROR GOTO vocab_error
        LoadDefaultVocabulary vocab_source
        success = 1
        GOTO vocab_done

vocab_error:
        success = 0

vocab_done:
        ON ERROR GOTO 0
        IF success = 0 THEN
            PRINT "Error: could not load vocabulary from "; vocab_source
            RETURN 0
        END IF
    ELSE
        PRINT "Using built-in byte-level vocabulary."
    END IF

    IF GPT2BasicLoadModel(MODEL_PATH) = 0 THEN
        PRINT "Error: trained GPT2-BASIC model files were not found in "; MODEL_PATH
        PRINT "Expected MODEL\GPT2CFG.TXT, MODEL\GPT2FX.BIN, MODEL\GPT2EXP.BIN,"
        PRINT "and MODEL\VOCAB.BIN for lexicon checkpoints."
        RETURN 0
    END IF

    IF g_tokenizer.vocab_size <> GPT2BasicVocabSize() THEN
        PRINT "Error: tokenizer vocab size "; g_tokenizer.vocab_size; _
              " does not match model vocab size "; GPT2BasicVocabSize()
        GPT2BasicFreeModel
        RETURN 0
    END IF

    g_model_initialized = 1
    PRINT "Production model initialization complete."
    PrintActiveModelInfo
    RETURN 1
END FUNCTION

SUB ShutdownModel()
    IF g_model_initialized = 0 THEN RETURN
    GPT2BasicFreeModel
    g_model_initialized = 0
END SUB

SUB PrintActiveModelInfo()
    IF GPT2BasicIsLoaded() = 0 THEN
        PRINT "  Model       : not loaded"
        RETURN
    END IF

    PRINT "  Build       : production fixed-point runtime"
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

FUNCTION GenerateNextToken(context() AS INTEGER, context_len AS INTEGER, temperature AS SINGLE, top_p AS SINGLE, top_k AS INTEGER) AS INTEGER
    IF GPT2BasicIsLoaded() = 0 THEN RETURN 0
    RETURN GPT2BasicNextToken(context(), context_len, temperature, top_p, top_k)
END FUNCTION

SUB GenerateText(prompt AS STRING, max_length AS INTEGER, temperature AS SINGLE, top_p AS SINGLE, top_k AS INTEGER)
    DIM input_tokens() AS INTEGER
    DIM input_token_count AS INTEGER
    DIM output_tokens() AS INTEGER
    DIM output_token_count AS INTEGER
    DIM i AS INTEGER
    DIM next_token AS INTEGER
    DIM generated_text AS STRING
    DIM start_time AS DOUBLE
    DIM end_time AS DOUBLE
    DIM total_time AS DOUBLE
    DIM tokens_per_second AS DOUBLE

    IF g_model_initialized = 0 THEN
        IF InitializeModel() = 0 THEN
            PRINT "Failed to initialize production model. Cannot generate text."
            RETURN
        END IF
    END IF

    Encode prompt, input_tokens(), input_token_count
    StripTrailingEOT input_tokens(), input_token_count

    IF input_token_count > GPT2BasicContextLength() THEN
        PRINT "Warning: prompt is too long ("; input_token_count; " tokens)."
        PRINT "Truncating to "; GPT2BasicContextLength(); " tokens."
        input_token_count = GPT2BasicContextLength()
    END IF

    IF input_token_count < 1 THEN
        REDIM input_tokens(0 TO 0)
        input_tokens(0) = 0
        input_token_count = 1
    END IF

    REDIM output_tokens(0 TO input_token_count + max_length - 1)
    FOR i = 0 TO input_token_count - 1
        output_tokens(i) = input_tokens(i)
    NEXT i
    output_token_count = input_token_count

    GPT2BasicBeginGeneration input_token_count
    start_time = TIMER

    FOR i = 0 TO max_length - 1
        next_token = GenerateNextToken(output_tokens(), output_token_count, temperature, top_p, top_k)
        output_tokens(output_token_count) = next_token
        output_token_count = output_token_count + 1

        IF next_token = 0 THEN EXIT FOR
        IF i >= SENTENCE_STOP_MIN_TOKENS THEN
            IF TinyGPTTokenEndsSentence(next_token) <> 0 THEN EXIT FOR
        END IF
    NEXT i

    end_time = TIMER
    total_time = end_time - start_time
    IF total_time < 0.0 THEN total_time = total_time + 86400.0
    IF total_time > 0.0 THEN
        tokens_per_second = (output_token_count - input_token_count) / total_time
    ELSE
        tokens_per_second = 0.0
    END IF

    PRINT "Generated "; (output_token_count - input_token_count); " tokens in "; _
          CompatFormat(total_time, "0.00"); " seconds ("; _
          CompatFormat(tokens_per_second, "0.00"); " tokens/sec)"

    generated_text = Decode(output_tokens(), output_token_count)
    PRINT
    PRINT "Generated Text:"
    PRINT "--------------"
    PRINT generated_text
    PRINT "--------------"
    PRINT "Runtime memory: "; GPT2BasicRuntimeMemoryBytes(); " bytes"
    PRINT "Tracked peak  : "; g_prod_peak_memory; " bytes"
END SUB

SUB RunTraceMode(prompt AS STRING, max_length AS INTEGER)
    DIM input_tokens() AS INTEGER
    DIM input_token_count AS INTEGER
    DIM context_tokens() AS INTEGER
    DIM context_len AS INTEGER
    DIM generated_tokens AS INTEGER
    DIM i AS INTEGER
    DIM next_token AS INTEGER
    DIM token_text AS STRING
    DIM generated_text AS STRING

    IF InitializeModel() = 0 THEN
        PRINT "TRACE_FAILED|stage=initialize_model"
        RETURN
    END IF

    Encode prompt, input_tokens(), input_token_count
    StripTrailingEOT input_tokens(), input_token_count
    IF input_token_count > GPT2BasicContextLength() THEN input_token_count = GPT2BasicContextLength()
    IF input_token_count < 1 THEN
        REDIM input_tokens(0 TO 0)
        input_tokens(0) = 0
        input_token_count = 1
    END IF

    REDIM context_tokens(0 TO input_token_count + max_length - 1)
    FOR i = 0 TO input_token_count - 1
        context_tokens(i) = input_tokens(i)
    NEXT i
    context_len = input_token_count

    PRINT "TRACE_BEGIN|suite=gpt2-basic-step|version=1"
    PRINT "TRACE_MODEL|profile=" + GPT2BasicProfileName() + _
          "|layers=" + ProdLongText(CLng(GPT2BasicLayerCount())) + _
          "|emb=" + ProdLongText(CLng(GPT2BasicEmbeddingDim())) + _
          "|heads=" + ProdLongText(CLng(GPT2BasicHeadCount())) + _
          "|ctx=" + ProdLongText(CLng(GPT2BasicContextLength())) + _
          "|vocab=" + ProdLongText(CLng(GPT2BasicVocabSize())) + _
          "|params=" + ProdLongText(GPT2BasicParameterCount()) + _
          "|runtime_bytes=" + ProdLongText(GPT2BasicRuntimeMemoryBytes())
    PRINT "TRACE_TOKENIZER|mode=" + TraceTokenizerModeName(g_tokenizer.tokenizer_mode) + _
          "|vocab=" + ProdLongText(CLng(g_tokenizer.vocab_size)) + _
          "|merges=" + ProdLongText(CLng(g_tokenizer.merge_count)) + _
          "|max_token_len=" + ProdLongText(CLng(g_tokenizer.max_token_length))
    PRINT "TRACE_PROMPT|text=" + TraceSafeText(prompt)
    PRINT "TRACE_PROMPT_TOKENS|count=" + ProdLongText(CLng(input_token_count))

    FOR i = 0 TO input_token_count - 1
        token_text = TraceSafeText(TinyGPTTokenText(input_tokens(i)))
        PRINT "TRACE_INPUT_TOKEN|pos=" + ProdLongText(CLng(i)) + _
              "|id=" + ProdLongText(CLng(input_tokens(i))) + _
              "|text=" + token_text
    NEXT i

    GPT2BasicBeginGeneration input_token_count
    PRINT "TRACE_STAGE|name=decode_begin|prompt_tokens=" + ProdLongText(CLng(input_token_count)) + _
          "|max_new_tokens=" + ProdLongText(CLng(max_length)) + _
          "|sampling=greedy_temperature_0"

    FOR i = 0 TO max_length - 1
        PRINT "TRACE_STAGE|name=forward_sample|step=" + ProdLongText(CLng(i)) + _
              "|context_len=" + ProdLongText(CLng(context_len))
        next_token = GenerateNextToken(context_tokens(), context_len, 0.0, DEFAULT_TOP_P, DEFAULT_TOP_K)
        context_tokens(context_len) = next_token
        context_len = context_len + 1
        generated_tokens = generated_tokens + 1
        token_text = TraceSafeText(TinyGPTTokenText(next_token))
        PRINT "TRACE_STEP|step=" + ProdLongText(CLng(i)) + _
              "|token=" + ProdLongText(CLng(next_token)) + _
              "|text=" + token_text + _
              "|ends_sentence=" + ProdLongText(CLng(TinyGPTTokenEndsSentence(next_token)))

        IF next_token = 0 THEN EXIT FOR
        IF i >= SENTENCE_STOP_MIN_TOKENS THEN
            IF TinyGPTTokenEndsSentence(next_token) <> 0 THEN EXIT FOR
        END IF
    NEXT i

    generated_text = Decode(context_tokens(), context_len)
    PRINT "TRACE_DECODED|text=" + TraceSafeText(generated_text)
    PRINT "TRACE_END|generated_tokens=" + ProdLongText(CLng(generated_tokens)) + _
          "|final_context_len=" + ProdLongText(CLng(context_len))
END SUB

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
    IF input_token_count > GPT2BasicContextLength() THEN input_token_count = GPT2BasicContextLength()
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
        next_token = GenerateNextToken(output_tokens(), output_token_count, 0.0, DEFAULT_TOP_P, DEFAULT_TOP_K)
        output_tokens(output_token_count) = next_token
        output_token_count = output_token_count + 1
        last_token = next_token

        IF next_token = 0 THEN EXIT FOR
        IF i >= SENTENCE_STOP_MIN_TOKENS THEN
            IF TinyGPTTokenEndsSentence(next_token) <> 0 THEN EXIT FOR
        END IF
    NEXT i

    end_time = TIMER
    elapsed = end_time - start_time
    IF elapsed < 0.0 THEN elapsed = elapsed + 86400.0

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
          "|prompt_tokens=" + ProdLongText(CLng(input_token_count)) + _
          "|generated_tokens=" + ProdLongText(CLng(generated_tokens)) + _
          "|seconds=" + ProdDoubleText(elapsed) + _
          "|tokens_per_sec=" + ProdDoubleText(tokens_per_second) + _
          "|last_token=" + ProdLongText(CLng(last_token))
END SUB

SUB RunHardwarePerformanceSuite()
    DIM total_runs AS INTEGER
    DIM total_tokens AS LONG
    DIM total_seconds AS DOUBLE
    DIM summary_tps AS DOUBLE
    DIM arithmetic AS STRING

    PRINT "PERF_BEGIN|suite=gpt2-basic-hardware|version=2"
    PRINT "PERF_BASIS|declared=emulation_or_physical|note=runner_records_qemu_or_pc_details"
    PRINT "PERF_CONTEXT|timed_region=decode_loop_only|sampling=greedy_temperature_0|console_progress=disabled|kv_cache=enabled|build=production"

    IF InitializeModel() = 0 THEN
        PRINT "PERF_FAILED|stage=initialize_model"
        PRINT "PERF_END"
        RETURN
    END IF

    IF GPT2BasicIsFixedPointLoaded() <> 0 THEN
        arithmetic = "q20.12_fixed"
    ELSE
        arithmetic = "float_reference"
    END IF

    PRINT "PERF_MACHINE|cpu_detected=not_probed_in_prod_build|cpu_enum=0|fpu=0|timer=freebasic_TIMER"
    PRINT "PERF_MODEL|profile=" + GPT2BasicProfileName() + _
          "|layers=" + ProdLongText(CLng(GPT2BasicLayerCount())) + _
          "|emb=" + ProdLongText(CLng(GPT2BasicEmbeddingDim())) + _
          "|heads=" + ProdLongText(CLng(GPT2BasicHeadCount())) + _
          "|ctx=" + ProdLongText(CLng(GPT2BasicContextLength())) + _
          "|vocab=" + ProdLongText(CLng(GPT2BasicVocabSize())) + _
          "|params=" + ProdLongText(GPT2BasicParameterCount()) + _
          "|fixed_bytes=" + ProdLongText(GPT2BasicFixedWeightBytes()) + _
          "|runtime_bytes=" + ProdLongText(GPT2BasicRuntimeMemoryBytes()) + _
          "|arithmetic=" + arithmetic

    IF g_prod_kernel_perf_request <> 0 THEN
        TinyGPTKernelPerfReset
        TinyGPTKernelPerfSetEnabled 1
    END IF

    RunHardwarePerformanceCase "real_inference", "What makes this real inference?", 90, total_runs, total_tokens, total_seconds
    RunHardwarePerformanceCase "486_target", "GPT2 BASIC on a 486", 90, total_runs, total_tokens, total_seconds
    RunHardwarePerformanceCase "basic_runtime", "A BASIC transformer runtime", 90, total_runs, total_tokens, total_seconds

    IF g_prod_kernel_perf_request <> 0 THEN
        TinyGPTKernelPerfSetEnabled 0
    END IF

    IF total_seconds > 0.0 THEN
        summary_tps = total_tokens / total_seconds
    ELSE
        summary_tps = 0.0
    END IF

    PRINT "PERF_SUMMARY|runs=" + ProdLongText(CLng(total_runs)) + _
          "|tokens=" + ProdLongText(total_tokens) + _
          "|seconds=" + ProdDoubleText(total_seconds) + _
          "|tokens_per_sec=" + ProdDoubleText(summary_tps)
    IF g_prod_kernel_perf_request <> 0 THEN TinyGPTKernelPerfReport
    PRINT "PERF_END"
END SUB

SUB InteractiveCompletion()
    DIM prompt AS STRING
    DIM prompt_line AS STRING

    ClearScreen
    PRINT "GPT2-BASIC production text completion"
    PRINT "Enter prompt text. Submit an empty line to generate."
    PRINT

    prompt = ""
    DO
        LINE INPUT prompt_line
        IF prompt_line = "" AND prompt <> "" THEN EXIT DO
        IF prompt <> "" THEN prompt = prompt + CHR$(13) + CHR$(10)
        prompt = prompt + prompt_line
    LOOP

    IF prompt <> "" THEN
        GenerateText prompt, DEFAULT_MAX_LENGTH, 0.0, DEFAULT_TOP_P, DEFAULT_TOP_K
        PRINT
        WaitForKeypress
    END IF
END SUB

SUB Main()
    DIM command_line AS STRING
    command_line = LCASE$(TRIM$(COMMAND$))

    IF command_line = "--demo" THEN
        RANDOMIZE 486
        ClearScreen
        PRINT "GPT2-BASIC production trained-model demo"
        PRINT
        GenerateText "What makes this real inference?", 90, 0.0, DEFAULT_TOP_P, DEFAULT_TOP_K
        ShutdownModel
        RETURN
    END IF

    IF command_line = "--quality" OR command_line = "--suite" THEN
        RANDOMIZE 486
        ClearScreen
        PRINT "GPT2-BASIC production runtime-regression quality suite"
        PRINT
        RunAutomatedQualitySuite "runtime-regression"
        ShutdownModel
        RETURN
    END IF

    IF command_line = "--quality-heldout" OR command_line = "--heldout-quality" THEN
        RANDOMIZE 486
        ClearScreen
        PRINT "GPT2-BASIC production held-out quality suite"
        PRINT
        RunAutomatedQualitySuite "heldout"
        ShutdownModel
        RETURN
    END IF

    IF command_line = "--quality-all" OR command_line = "--suite-all" THEN
        RANDOMIZE 486
        ClearScreen
        PRINT "GPT2-BASIC production quality suites"
        PRINT
        RunAutomatedQualitySuite "all"
        ShutdownModel
        RETURN
    END IF

    IF command_line = "--perf" OR command_line = "--hardware-perf" THEN
        RANDOMIZE 486
        g_prod_kernel_perf_request = 0
        RunHardwarePerformanceSuite
        ShutdownModel
        RETURN
    END IF

    IF command_line = "--kernel-perf" THEN
        RANDOMIZE 486
        g_prod_kernel_perf_request = 1
        RunHardwarePerformanceSuite
        g_prod_kernel_perf_request = 0
        ShutdownModel
        RETURN
    END IF

    IF command_line = "--trace" OR command_line = "--step-trace" OR command_line = "--educational-trace" THEN
        RANDOMIZE 486
        RunTraceMode "What makes this real inference?", 12
        ShutdownModel
        RETURN
    END IF

    IF command_line = "--vectors" OR command_line = "--parity" THEN
        RANDOMIZE 486
        ClearScreen
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
        ShutdownModel
        RETURN
    END IF

    RANDOMIZE TIMER
    InteractiveCompletion
    ShutdownModel
END SUB

Main
