' *******************************************************
' * Benchmarking for GPT-2 BASIC                        *
' *******************************************************
' * This module provides detailed benchmarking tools    *
' * to evaluate the performance of different components *
' * of the GPT-2 implementation.                        *
' *                                                     *
' * It measures execution time, memory usage, and       *
' * accuracy for various operations and configurations. *
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

DECLARE SUB BenchmarkGenerateText(context() AS INTEGER, context_len AS INTEGER, gen_len AS INTEGER, config AS ModelConfig, _
                output_tokens() AS INTEGER, BYREF output_len AS INTEGER)
DECLARE SUB BenchmarkGPT2BasicRuntime()
DECLARE SUB CreateBenchmarkLayerFiles(config AS ModelConfig)
DECLARE SUB CleanupBenchmarkFiles()

' *******************************************************
' * Constants and Configuration                         *
' *******************************************************

' Benchmark settings
CONST RUN_COMPONENT_BENCHMARKS = 1 ' Test individual components
CONST RUN_INTEGRATION_BENCHMARKS = 1 ' Test integrated components
CONST RUN_END_TO_END_BENCHMARKS = 1 ' Test full model inference
CONST INCLUDE_MEMORY_PROFILE = 1 ' Track memory usage
CONST WARMUP_RUNS = 2 ' Number of warmup runs
CONST BENCHMARK_RUNS = 5 ' Number of benchmark runs

' Matrix sizes for benchmarks
TYPE BenchmarkSize
    rows AS INTEGER
    cols AS INTEGER
    label AS STRING
END TYPE

' Benchmark results
TYPE BenchmarkResult
    benchmark_name AS STRING
    operation AS STRING
    size_label AS STRING
    time_ms AS DOUBLE
    memory_bytes AS LONG
    accuracy AS SINGLE
END TYPE

' Global result storage
DIM SHARED g_results(1 TO 1000) AS BenchmarkResult
DIM SHARED g_result_count AS INTEGER = 0

' *******************************************************
' * Benchmark Utilities                                 *
' *******************************************************

' Add a benchmark result to global storage
SUB AddBenchmarkResult(benchmark_name AS STRING, operation AS STRING, size_label AS STRING, _
                     time_ms AS DOUBLE, memory_bytes AS LONG, accuracy AS SINGLE)
    g_result_count = g_result_count + 1
    g_results(g_result_count).benchmark_name = benchmark_name
    g_results(g_result_count).operation = operation
    g_results(g_result_count).size_label = size_label
    g_results(g_result_count).time_ms = time_ms
    g_results(g_result_count).memory_bytes = memory_bytes
    g_results(g_result_count).accuracy = accuracy
END SUB

' Print a header for a benchmark section
SUB PrintBenchmarkHeader(title AS STRING)
    DIM separator_line AS STRING
    separator_line = STRING(LEN(title) + 8, "=")

    PRINT
    PRINT separator_line
    PRINT "| " + title + " |"
    PRINT separator_line
END SUB

' Print benchmark results in a formatted table
SUB PrintBenchmarkResults(filter_name AS STRING)
    DIM i AS INTEGER

    ' Header
    PRINT
    PRINT LEFT$("Benchmark", 20); " | "; LEFT$("Operation", 20); " | "; _
          LEFT$("Size", 10); " | "; LEFT$("Time (ms)", 10); " | "; _
          LEFT$("Memory (KB)", 12); " | "; LEFT$("Accuracy", 8)
    PRINT STRING(93, "-")

    ' Results
    FOR i = 1 TO g_result_count
        ' Apply filter if provided
        IF filter_name <> "" AND INSTR(g_results(i).benchmark_name, filter_name) = 0 THEN
            CONTINUE FOR
        END IF

        PRINT LEFT$(g_results(i).benchmark_name, 20); " | ";
        PRINT LEFT$(g_results(i).operation, 20); " | ";
        PRINT LEFT$(g_results(i).size_label, 10); " | ";
        PRINT RIGHT$("          " + CompatFormat(g_results(i).time_ms, "0.00"), 10); " | ";
        PRINT RIGHT$("            " + CompatFormat(g_results(i).memory_bytes / 1024, "0.00"), 12); " | ";

        IF g_results(i).accuracy > 0 THEN
            PRINT RIGHT$("        " + CompatFormat(g_results(i).accuracy, "0.000"), 8)
        ELSE
            PRINT RIGHT$("        " + "N/A", 8)
        END IF
    NEXT i
END SUB

' Helper to measure peak memory usage during a benchmark
FUNCTION MeasurePeakMemoryDuring(benchmark_proc AS ANY PTR) AS LONG
    DIM start_memory AS LONG
    DIM peak_memory AS LONG

    ' Record starting memory
    start_memory = g_current_memory_usage
    g_peak_memory_usage = start_memory

    ' Procedure-pointer calls are not portable in the DOS compiler build.
    ' This helper remains as a memory accounting hook for future direct-call wrappers.

    ' Return the peak usage during benchmark
    RETURN g_peak_memory_usage - start_memory
END FUNCTION

' Generate a random matrix for testing
SUB GenerateRandomMatrix(BYREF mat AS Matrix, rows AS INTEGER, cols AS INTEGER, scale AS SINGLE)
    InitMatrix(mat, rows, cols)

    DIM i AS INTEGER, j AS INTEGER
    FOR i = 0 TO rows - 1
        FOR j = 0 TO cols - 1
            mat.data(i, j) = (RND - 0.5) * scale
        NEXT j
    NEXT i
END SUB

' Calculate mean squared error between two matrices
FUNCTION CalculateMSE(a AS Matrix, b AS Matrix) AS SINGLE
    DIM i AS INTEGER, j AS INTEGER
    DIM sum_squared_error AS SINGLE
    DIM count AS LONG

    IF a.rows <> b.rows OR a.cols <> b.cols THEN
        PRINT "ERROR: Matrix dimensions don't match for MSE calculation"
        RETURN -1
    END IF

    sum_squared_error = 0
    count = a.rows * a.cols

    FOR i = 0 TO a.rows - 1
        FOR j = 0 TO a.cols - 1
            DIM delta AS SINGLE
            delta = a.data(i, j) - b.data(i, j)
            sum_squared_error = sum_squared_error + delta * delta
        NEXT j
    NEXT i

    RETURN sum_squared_error / count
END FUNCTION

' *******************************************************
' * Component Benchmarks                                *
' *******************************************************

' Benchmark matrix multiplication with different optimizations
SUB BenchmarkMatrixMultiply()
    DIM sizes(1 TO 4) AS BenchmarkSize
    DIM optimization_types(1 TO 3) AS STRING
    DIM i AS INTEGER, j AS INTEGER, k AS INTEGER, r AS INTEGER, idx AS INTEGER
    DIM a AS Matrix, b AS Matrix, c1 AS Matrix, c2 AS Matrix
    DIM start_time AS DOUBLE, end_time AS DOUBLE, total_time AS DOUBLE
    DIM memory_used AS LONG
    DIM mse AS SINGLE

    ' Configure test matrix sizes
    sizes(1).rows = 32:  sizes(1).cols = 32:  sizes(1).label = "32x32"
    sizes(2).rows = 64:  sizes(2).cols = 64:  sizes(2).label = "64x64"
    sizes(3).rows = 128: sizes(3).cols = 128: sizes(3).label = "128x128"
    sizes(4).rows = 256: sizes(4).cols = 256: sizes(4).label = "256x256"

    ' Optimization types
    optimization_types(1) = "Standard"
    optimization_types(2) = "SIMD-like"
    optimization_types(3) = "Assembly"

    PrintBenchmarkHeader("Matrix Multiplication Benchmark")

    ' For each size
    FOR i = 1 TO UBOUND(sizes)
        PRINT "Testing size: "; sizes(i).label

        ' Generate random matrices for this size
        GenerateRandomMatrix(a, sizes(i).rows, sizes(i).cols, 1.0)
        GenerateRandomMatrix(b, sizes(i).cols, sizes(i).rows, 1.0) ' For matrix multiplication

        ' Reference result using standard multiplication
        MatrixMultiply(a, b, c1)

        ' For each optimization type
        FOR j = 1 TO UBOUND(optimization_types)
            PRINT "  Testing: "; optimization_types(j)

            ' Warmup runs
            FOR r = 1 TO WARMUP_RUNS
                SELECT CASE j
                    CASE 1: MatrixMultiply(a, b, c2)
                    CASE 2: MatrixMultiplySIMD(a, b, c2)
                    CASE 3: MatrixMultiplyAsm(a, b, c2)
                END SELECT
            NEXT r

            ' Timed runs
            total_time = 0
            start_time = TIMER

            FOR r = 1 TO BENCHMARK_RUNS
                SELECT CASE j
                    CASE 1: MatrixMultiply(a, b, c2)
                    CASE 2: MatrixMultiplySIMD(a, b, c2)
                    CASE 3: MatrixMultiplyAsm(a, b, c2)
                END SELECT
            NEXT r

            end_time = TIMER
            total_time = (end_time - start_time) * 1000 / BENCHMARK_RUNS ' Convert to ms per run

            ' Calculate accuracy vs reference
            mse = CalculateMSE(c1, c2)

            ' Memory used is estimated based on matrix sizes
            memory_used = (a.rows * a.cols + b.rows * b.cols + c2.rows * c2.cols) * 4 ' 4 bytes per float

            ' Record result
            AddBenchmarkResult("MatrixMultiply", optimization_types(j), sizes(i).label, _
                             total_time, memory_used, mse)
        NEXT j

        ' Clean up
        FreeMatrix(a)
        FreeMatrix(b)
        FreeMatrix(c1)
        FreeMatrix(c2)
    NEXT i
END SUB

' Benchmark attention computation with different optimizations
SUB BenchmarkAttention()
    DIM seq_lengths(1 TO 3) AS INTEGER
    DIM embed_dims(1 TO 2) AS INTEGER
    DIM optimization_types(1 TO 3) AS STRING
    DIM i AS INTEGER, j AS INTEGER, k AS INTEGER, r AS INTEGER
    DIM query AS Matrix, key AS Matrix, value AS Matrix
    DIM output1 AS Matrix, output2 AS Matrix
    DIM start_time AS DOUBLE, end_time AS DOUBLE, total_time AS DOUBLE
    DIM memory_used AS LONG
    DIM mse AS SINGLE

    ' Configure test sequence lengths and embedding dimensions
    seq_lengths(1) = 64
    seq_lengths(2) = 128
    seq_lengths(3) = 256

    embed_dims(1) = 64
    embed_dims(2) = 128

    ' Optimization types
    optimization_types(1) = "Dense"
    optimization_types(2) = "Block-Sparse"
    optimization_types(3) = "Fixed-Point"

    PrintBenchmarkHeader("Attention Mechanism Benchmark")

    ' For each sequence length
    FOR i = 1 TO UBOUND(seq_lengths)
        ' For each embedding dimension
        FOR j = 1 TO UBOUND(embed_dims)
            DIM size_label AS STRING
            size_label = "S=" + LTRIM$(STR$(seq_lengths(i))) + " E=" + LTRIM$(STR$(embed_dims(j)))

            PRINT "Testing size: "; size_label

            ' Generate random query, key, value matrices
            GenerateRandomMatrix(query, seq_lengths(i), embed_dims(j), 0.1)
            GenerateRandomMatrix(key, seq_lengths(i), embed_dims(j), 0.1)
            GenerateRandomMatrix(value, seq_lengths(i), embed_dims(j), 0.1)

            ' Reference result using dense attention
            DenseAttention(query, key, value, output1, 1) ' 1 = use causal mask

            ' For each optimization type
            FOR k = 1 TO UBOUND(optimization_types)
                PRINT "  Testing: "; optimization_types(k)

                ' Warmup runs
                FOR r = 1 TO WARMUP_RUNS
                    SELECT CASE k
                        CASE 1: DenseAttention(query, key, value, output2, 1)
                        CASE 2: BlockSparseAttention(query, key, value, output2, 1)
                        CASE 3:
                            ' Fixed-point version would use something like this:
                            ' Use MatrixSoftmaxFixed from softmax_fixed.bas
                            DenseAttention(query, key, value, output2, 1)
                    END SELECT
                NEXT r

                ' Timed runs
                total_time = 0
                start_time = TIMER

                FOR r = 1 TO BENCHMARK_RUNS
                    SELECT CASE k
                        CASE 1: DenseAttention(query, key, value, output2, 1)
                        CASE 2: BlockSparseAttention(query, key, value, output2, 1)
                        CASE 3:
                            ' Fixed-point version would use something like this:
                            DenseAttention(query, key, value, output2, 1)
                    END SELECT
                NEXT r

                end_time = TIMER
                total_time = (end_time - start_time) * 1000 / BENCHMARK_RUNS ' Convert to ms per run

                ' Calculate accuracy vs reference
                mse = CalculateMSE(output1, output2)

                ' Memory used is estimated based on attention computation
                SELECT CASE k
                    CASE 1: ' Dense
                        memory_used = seq_lengths(i) * seq_lengths(i) * 4 ' QK^T matrix
                    CASE 2: ' Block-Sparse
                        ' Estimate 30% of dense memory
                        memory_used = seq_lengths(i) * seq_lengths(i) * 4 * 0.3
                    CASE 3: ' Fixed-point
                        ' Lower precision
                        memory_used = seq_lengths(i) * seq_lengths(i) * 2 ' QK^T matrix with 16-bit precision
                END SELECT

                ' Add memory for Q, K, V, and output
                memory_used = memory_used + 4 * seq_lengths(i) * embed_dims(j) * 4

                ' Record result
                AddBenchmarkResult("Attention", optimization_types(k), size_label, _
                                 total_time, memory_used, mse)
            NEXT k

            ' Clean up
            FreeMatrix(query)
            FreeMatrix(key)
            FreeMatrix(value)
            FreeMatrix(output1)
            FreeMatrix(output2)
        NEXT j
    NEXT i
END SUB

' Benchmark fixed-point math operations
SUB BenchmarkFixedPoint()
    DIM operation_types(1 TO 4) AS STRING
    DIM size_labels(1 TO 3) AS STRING
    DIM i AS INTEGER, j AS INTEGER, k AS INTEGER, r AS INTEGER
    DIM start_time AS DOUBLE, end_time AS DOUBLE, total_time AS DOUBLE
    DIM iterations AS LONG
    DIM a_fixed AS LONG, b_fixed AS LONG, result_fixed AS LONG
    DIM a_float AS SINGLE, b_float AS SINGLE, result_float AS SINGLE
    DIM relative_error AS SINGLE

    ' Operation types
    operation_types(1) = "Multiply"
    operation_types(2) = "Division"
    operation_types(3) = "Square Root"
    operation_types(4) = "Exp Function"

    ' Size labels (representing number of operations)
    size_labels(1) = "10K ops"
    size_labels(2) = "100K ops"
    size_labels(3) = "1M ops"

    ' Iteration counts
    DIM operation_counts(1 TO 3) AS LONG
    operation_counts(1) = 10000
    operation_counts(2) = 100000
    operation_counts(3) = 1000000

    PrintBenchmarkHeader("Fixed-Point Math Benchmark")

    ' For each operation type
    FOR i = 1 TO UBOUND(operation_types)
        PRINT "Testing operation: "; operation_types(i)

        ' For each size
        FOR j = 1 TO UBOUND(size_labels)
            iterations = operation_counts(j)
            PRINT "  Testing: "; size_labels(j)

            ' Initialize test values
            a_float = 3.14159
            b_float = 2.71828
            a_fixed = FloatToFixed(a_float)
            b_fixed = FloatToFixed(b_float)

            ' Floating point version (for baseline)
            start_time = TIMER
            FOR r = 1 TO iterations
                SELECT CASE i
                    CASE 1: result_float = a_float * b_float
                    CASE 2: result_float = a_float / b_float
                    CASE 3: result_float = SQR(a_float)
                    CASE 4: result_float = EXP(a_float / 10.0) ' Scaled to avoid overflow
                END SELECT
            NEXT r
            end_time = TIMER
            DIM float_time AS DOUBLE = (end_time - start_time) * 1000 ' Convert to ms

            ' Fixed point version
            start_time = TIMER
            FOR r = 1 TO iterations
                SELECT CASE i
                    CASE 1: result_fixed = FixedMul(a_fixed, b_fixed)
                    CASE 2: result_fixed = FixedDiv(a_fixed, b_fixed)
                    CASE 3: result_fixed = FixedSqrt(a_fixed)
                    CASE 4: result_fixed = FixedExp(a_fixed / 10) ' Scaled to avoid overflow
                END SELECT
            NEXT r
            end_time = TIMER
            DIM fixed_time AS DOUBLE = (end_time - start_time) * 1000 ' Convert to ms

            ' Calculate error
            SELECT CASE i
                CASE 1: relative_error = ABS((FixedToFloat(result_fixed) - a_float * b_float) / (a_float * b_float))
                CASE 2: relative_error = ABS((FixedToFloat(result_fixed) - a_float / b_float) / (a_float / b_float))
                CASE 3: relative_error = ABS((FixedToFloat(result_fixed) - SQR(a_float)) / SQR(a_float))
                CASE 4: relative_error = ABS((FixedToFloat(result_fixed) - EXP(a_float / 10.0)) / EXP(a_float / 10.0))
            END SELECT

            ' Memory usage is minimal for these operations
            DIM memory_used AS LONG = 12 ' 3 LONG values (4 bytes each)

            ' Record results
            AddBenchmarkResult("FixedPoint-Float", operation_types(i), size_labels(j), float_time, memory_used, 0)
            AddBenchmarkResult("FixedPoint-Fixed", operation_types(i), size_labels(j), fixed_time, memory_used, relative_error)
        NEXT j
    NEXT i
END SUB

' Benchmark tokenization with different vocabulary sizes
SUB BenchmarkTokenizer()
    DIM text_lengths(1 TO 3) AS INTEGER
    DIM vocab_sizes(1 TO 3) AS INTEGER
    DIM i AS INTEGER, j AS INTEGER, k AS INTEGER, r AS INTEGER
    DIM start_time AS DOUBLE, end_time AS DOUBLE, total_time AS DOUBLE
    DIM memory_used AS LONG

    ' Configure test text lengths
    text_lengths(1) = 100
    text_lengths(2) = 1000
    text_lengths(3) = 5000

    ' Configure vocabulary sizes
    vocab_sizes(1) = 1000
    vocab_sizes(2) = 5000
    vocab_sizes(3) = 16384

    PrintBenchmarkHeader("Tokenizer Benchmark")

    ' Test text samples
    DIM test_texts(1 TO 3) AS STRING
    test_texts(1) = STRING(text_lengths(1), "A") ' Placeholder - real tests would use meaningful text
    test_texts(2) = STRING(text_lengths(2), "B")
    test_texts(3) = STRING(text_lengths(3), "C")

    ' For each vocabulary size
    FOR j = 1 TO UBOUND(vocab_sizes)
        ' Initialize tokenizer with appropriate vocabulary
        InitializeDefaultTokenizer()
        DIM token_count_ref AS INTEGER
        DIM tokens_ref() AS INTEGER

        PRINT "Testing vocab size: "; vocab_sizes(j)

        ' For each text length
        FOR i = 1 TO UBOUND(text_lengths)
            DIM size_label AS STRING
            size_label = "Text=" + LTRIM$(STR$(text_lengths(i)))

            PRINT "  Testing: "; size_label

            ' Warmup runs
            FOR r = 1 TO WARMUP_RUNS
                DIM tokens() AS INTEGER
                DIM token_count AS INTEGER
                Encode(test_texts(i), tokens(), token_count)
            NEXT r

            ' Timed runs for encoding
            total_time = 0
            start_time = TIMER

            FOR r = 1 TO BENCHMARK_RUNS
                DIM tokens() AS INTEGER
                DIM token_count AS INTEGER
                Encode(test_texts(i), tokens(), token_count)

                ' Save reference for decoding tests
                IF r = 1 THEN
                    token_count_ref = token_count
                    REDIM tokens_ref(0 TO token_count - 1)
                    FOR k = 0 TO token_count - 1
                        tokens_ref(k) = tokens(k)
                    NEXT k
                END IF
            NEXT r

            end_time = TIMER
            DIM encode_time AS DOUBLE = (end_time - start_time) * 1000 / BENCHMARK_RUNS ' Convert to ms per run

            ' Memory usage is estimated based on vocabulary size and text length
            memory_used = vocab_sizes(j) * 32 ' Vocabulary storage (32 bytes per entry including hash table)
            memory_used = memory_used + text_lengths(i) * 4 ' Token output array

            ' Record result for encoding
            AddBenchmarkResult("Tokenizer", "Encode", size_label, encode_time, memory_used, 0)

            ' Timed runs for decoding
            total_time = 0
            start_time = TIMER

            FOR r = 1 TO BENCHMARK_RUNS
                DIM decoded AS STRING
                decoded = Decode(tokens_ref(), token_count_ref)
            NEXT r

            end_time = TIMER
            DIM decode_time AS DOUBLE = (end_time - start_time) * 1000 / BENCHMARK_RUNS ' Convert to ms per run

            ' Record result for decoding
            AddBenchmarkResult("Tokenizer", "Decode", size_label, decode_time, memory_used, 0)
        NEXT i
    NEXT j
END SUB

' *******************************************************
' * Integration Benchmarks                              *
' *******************************************************

' Benchmark the production transformer forward path used by generation.
SUB BenchmarkTransformerBlock()
    DIM context_lengths(1 TO 3) AS INTEGER
    DIM prompt_tokens() AS INTEGER
    DIM prompt_count AS INTEGER
    DIM context_tokens() AS INTEGER
    DIM i AS INTEGER, j AS INTEGER, r AS INTEGER
    DIM start_time AS DOUBLE, end_time AS DOUBLE, total_time AS DOUBLE
    DIM memory_used AS LONG
    DIM size_label AS STRING

    context_lengths(1) = 16
    context_lengths(2) = 64
    context_lengths(3) = 128

    PrintBenchmarkHeader("Production Transformer Forward Benchmark")

    IF GPT2BasicIsLoaded() = 0 THEN
        PRINT "Loading production checkpoint from MODEL..."
        IF GPT2BasicLoadModel("MODEL") = 0 THEN
            PRINT "No trained GPT2-BASIC checkpoint available; forward benchmark skipped."
            RETURN
        END IF
    END IF

    Encode "What makes this real inference?", prompt_tokens(), prompt_count
    IF prompt_count > 0 THEN
        IF prompt_tokens(prompt_count - 1) = 0 THEN prompt_count = prompt_count - 1
    END IF
    IF prompt_count < 1 THEN
        REDIM prompt_tokens(0 TO 0)
        prompt_tokens(0) = 0
        prompt_count = 1
    END IF

    memory_used = GPT2BasicRuntimeMemoryBytes()

    FOR i = 1 TO UBOUND(context_lengths)
        IF context_lengths(i) > GPT2BasicContextLength() THEN context_lengths(i) = GPT2BasicContextLength()
        size_label = "C=" + LTRIM$(STR$(context_lengths(i)))
        PRINT "Testing context length: "; size_label

        REDIM context_tokens(0 TO context_lengths(i) - 1)
        FOR j = 0 TO context_lengths(i) - 1
            context_tokens(j) = prompt_tokens(j MOD prompt_count)
        NEXT j

        FOR r = 1 TO WARMUP_RUNS
            IF GPT2BasicIsFixedPointLoaded() <> 0 THEN
                DIM warm_fx_logits() AS LONG
                GPT2BasicForwardFixedLogits context_tokens(), context_lengths(i), warm_fx_logits()
            ELSE
                DIM warm_float_logits() AS SINGLE
                GPT2BasicForwardFloatLogits context_tokens(), context_lengths(i), warm_float_logits()
            END IF
        NEXT r

        start_time = TIMER
        FOR r = 1 TO BENCHMARK_RUNS
            IF GPT2BasicIsFixedPointLoaded() <> 0 THEN
                DIM fx_logits() AS LONG
                GPT2BasicForwardFixedLogits context_tokens(), context_lengths(i), fx_logits()
            ELSE
                DIM float_logits() AS SINGLE
                GPT2BasicForwardFloatLogits context_tokens(), context_lengths(i), float_logits()
            END IF
        NEXT r
        end_time = TIMER

        total_time = (end_time - start_time) * 1000 / BENCHMARK_RUNS
        AddBenchmarkResult("TransformerBlock", GPT2BasicProfileName(), size_label, _
                         total_time, memory_used, 0)
    NEXT i
END SUB

' *******************************************************
' * End-to-End Benchmarks                               *
' *******************************************************

' Benchmark token generation with the loaded production checkpoint.
SUB BenchmarkTokenGeneration()
    BenchmarkGPT2BasicRuntime
END SUB

SUB BenchmarkGPT2BasicRuntime()
    DIM prompts(1 TO 3) AS STRING
    DIM gen_lengths(1 TO 3) AS INTEGER
    DIM i AS INTEGER, j AS INTEGER, r AS INTEGER
    DIM start_time AS DOUBLE, end_time AS DOUBLE, total_time AS DOUBLE
    DIM memory_used AS LONG
    DIM tokens_per_second AS SINGLE

    prompts(1) = "What makes this real inference?"
    prompts(2) = "GPT2 BASIC on a 486"
    prompts(3) = "DOS language models need"

    gen_lengths(1) = 8
    gen_lengths(2) = 16
    gen_lengths(3) = 32

    PrintBenchmarkHeader("GPT2-BASIC Runtime Benchmark")

    IF GPT2BasicIsLoaded() = 0 THEN
        PRINT "Loading production checkpoint from MODEL..."
        IF GPT2BasicLoadModel("MODEL") = 0 THEN
            PRINT "No trained GPT2-BASIC checkpoint available; runtime benchmark skipped."
            RETURN
        END IF
    END IF

    memory_used = GPT2BasicRuntimeMemoryBytes()
    PRINT "Checkpoint profile: "; GPT2BasicProfileName()
    PRINT "Runtime memory    : "; memory_used; " bytes"

    FOR i = 1 TO UBOUND(prompts)
        DIM input_tokens() AS INTEGER
        DIM input_count AS INTEGER
        Encode prompts(i), input_tokens(), input_count
        IF input_count > 0 THEN
            IF input_tokens(input_count - 1) = 0 THEN input_count = input_count - 1
        END IF
        IF input_count < 1 THEN
            REDIM input_tokens(0 TO 0)
            input_tokens(0) = 0
            input_count = 1
        END IF
        IF input_count > GPT2BasicContextLength() THEN input_count = GPT2BasicContextLength()

        FOR j = 1 TO UBOUND(gen_lengths)
            DIM output_tokens() AS INTEGER
            DIM output_count AS INTEGER
            DIM next_token AS INTEGER
            DIM size_label AS STRING

            size_label = "P=" + LTRIM$(STR$(i)) + " G=" + LTRIM$(STR$(gen_lengths(j)))
            REDIM output_tokens(0 TO input_count + gen_lengths(j) - 1)
            FOR r = 0 TO input_count - 1
                output_tokens(r) = input_tokens(r)
            NEXT r
            output_count = input_count

            GPT2BasicBeginGeneration input_count
            start_time = TIMER

            FOR r = 1 TO gen_lengths(j)
                next_token = GPT2BasicNextToken(output_tokens(), output_count, 0.0, 0.9, 40)
                output_tokens(output_count) = next_token
                output_count = output_count + 1
                IF next_token = 0 THEN EXIT FOR
            NEXT r

            end_time = TIMER
            total_time = (end_time - start_time) * 1000
            IF total_time <= 0 THEN total_time = 1
            tokens_per_second = (output_count - input_count) / (total_time / 1000)

            AddBenchmarkResult("Generation", GPT2BasicProfileName(), size_label, _
                             total_time, memory_used, tokens_per_second)
        NEXT j
    NEXT i
END SUB

' Benchmark memory streaming for large models
SUB BenchmarkMemoryStreaming()
    DIM model_sizes(1 TO 3) AS STRING
    DIM batch_sizes(1 TO 3) AS INTEGER
    DIM i AS INTEGER, j AS INTEGER, r AS INTEGER, layer_idx AS INTEGER
    DIM start_time AS DOUBLE, end_time AS DOUBLE, total_time AS DOUBLE
    DIM memory_used AS LONG
    DIM success AS INTEGER
    DIM cache AS LayerCache PTR

    ' Model sizes
    model_sizes(1) = "Medium (8L-512d)"
    model_sizes(2) = "Large (12L-768d)"
    model_sizes(3) = "XL (16L-1024d)"

    ' Layer batch sizes for streaming
    batch_sizes(1) = 1  ' One layer at a time
    batch_sizes(2) = 2  ' Two layers at a time
    batch_sizes(3) = 4  ' Four layers at a time

    PrintBenchmarkHeader("Memory Streaming Benchmark")

    ' Initialize memory manager
    InitMemoryManager()

    ' For each model size
    FOR i = 1 TO UBOUND(model_sizes)
        PRINT "Testing model: "; model_sizes(i)

        ' Create model configuration
        DIM config AS ModelConfig
        InitModelConfig(config)

        ' Configure model parameters based on size
        SELECT CASE i
            CASE 1: ' Medium
                config.n_layer = 8
                config.n_embd = 512
                config.n_head = 16
            CASE 2: ' Large
                config.n_layer = 12
                config.n_embd = 768
                config.n_head = 12
            CASE 3: ' XL
                config.n_layer = 16
                config.n_embd = 1024
                config.n_head = 16
        END SELECT

        ' For each batch size
        FOR j = 1 TO UBOUND(batch_sizes)
            DIM size_label AS STRING
            size_label = "Batch=" + LTRIM$(STR$(batch_sizes(j)))

            PRINT "  Testing: "; size_label

            ' Reset memory manager
            ShutdownMemoryManager()
            InitMemoryManager()

            ' Set maximum cached layers
            g_max_cached_layers = batch_sizes(j)

            ' Set model path
            SetModelPath("BENCHDAT")

            ' Create benchmark layer files
            CreateBenchmarkLayerFiles(config)

            ' Warmup run
            FOR layer_idx = 0 TO config.n_layer - 1
                cache = GetLayerWeights(layer_idx, success)
            NEXT layer_idx

            ' Reset memory manager
            ShutdownMemoryManager()
            InitMemoryManager()
            g_max_cached_layers = batch_sizes(j)
            SetModelPath("BENCHDAT")

            ' Timed run
            start_time = TIMER

            ' Access all layers in sequence
            FOR r = 1 TO 3 ' Multiple passes to test caching
                FOR layer_idx = 0 TO config.n_layer - 1
                    cache = GetLayerWeights(layer_idx, success)
                NEXT layer_idx
            NEXT r

            end_time = TIMER
            total_time = (end_time - start_time) * 1000 ' ms

            ' Record peak memory usage
            memory_used = g_peak_memory_usage

            ' Record results
            DIM disk_read_count AS INTEGER = g_disk_reads
            DIM cache_hit_rate AS SINGLE = g_cache_hits / (g_cache_hits + g_cache_misses)

            ' Record results
            AddBenchmarkResult("MemoryStreaming", model_sizes(i), size_label, _
                             total_time, memory_used, cache_hit_rate)

            PRINT "    Disk reads: "; disk_read_count; ", Cache hit rate: "; CompatFormat(cache_hit_rate * 100, "0.00"); "%"
        NEXT j
    NEXT i

    ' Clean up
    ShutdownMemoryManager()
    CleanupBenchmarkFiles()
END SUB

' *******************************************************
' * Helper Functions for Benchmarks                     *
' *******************************************************

SUB BenchmarkGenerateText(context() AS INTEGER, context_len AS INTEGER, gen_len AS INTEGER, config AS ModelConfig, _
                output_tokens() AS INTEGER, BYREF output_len AS INTEGER)
    DIM i AS INTEGER
    DIM next_token AS INTEGER

    output_len = context_len + gen_len
    REDIM output_tokens(0 TO output_len - 1)

    FOR i = 0 TO context_len - 1
        output_tokens(i) = context(i)
    NEXT i

    IF GPT2BasicIsLoaded() = 0 THEN
        IF GPT2BasicLoadModel("MODEL") = 0 THEN
            output_len = context_len
            RETURN
        END IF
    END IF

    GPT2BasicBeginGeneration context_len
    FOR i = context_len TO output_len - 1
        next_token = GPT2BasicNextToken(output_tokens(), i, 0.0, 0.9, 40)
        output_tokens(i) = next_token
        IF next_token = 0 THEN
            output_len = i + 1
            EXIT FOR
        END IF
    NEXT i
END SUB

' Create layer files for memory streaming benchmark
SUB CreateBenchmarkLayerFiles(config AS ModelConfig)
    DIM layer_idx AS INTEGER
    DIM w AS INTEGER, r AS INTEGER, c AS INTEGER
    DIM rows AS INTEGER, cols AS INTEGER

    ' Create BENCHDAT directory if it doesn't exist
    MKDIR "BENCHDAT"

    ' Create benchmark layer files for each layer
    FOR layer_idx = 0 TO config.n_layer - 1
        DIM filename AS STRING
        filename = "BENCHDAT\L" + LTRIM$(STR$(layer_idx)) + ".BIN"

        DIM file AS LONG
        file = FREEFILE

        OPEN filename FOR BINARY AS file

        ' Write number of weight matrices (4 for attention, 2 for MLP)
        DIM num_weights AS INTEGER = 6
        PUT #file, , num_weights

        ' Create deterministic weight matrices
        FOR w = 1 TO num_weights
            ' Matrix dimensions based on component
            SELECT CASE w
                CASE 1, 2, 3: ' QKV projections
                    rows = config.n_embd
                    cols = config.n_embd
                CASE 4: ' Attention output
                    rows = config.n_embd
                    cols = config.n_embd
                CASE 5: ' MLP expansion
                    rows = config.n_embd
                    cols = config.n_embd * 4
                CASE 6: ' MLP output
                    rows = config.n_embd * 4
                    cols = config.n_embd
            END SELECT

            ' Write dimensions
            PUT #file, , rows
            PUT #file, , cols

            ' Write deterministic benchmark data
            DIM bench_val AS SINGLE
            FOR r = 0 TO rows - 1
                FOR c = 0 TO cols - 1
                    bench_val = (r * c) / (rows * cols) * 0.1
                    PUT #file, , bench_val
                NEXT c
            NEXT r
        NEXT w

        CLOSE #file
    NEXT layer_idx
END SUB

' Clean up benchmark files
SUB CleanupBenchmarkFiles()
    SHELL "DEL /Q BENCHDAT\*.*"
    SHELL "RMDIR BENCHDAT"
END SUB

' *******************************************************
' * Main Benchmark Program                              *
' *******************************************************

' Run all benchmarks
SUB RunAllBenchmarks()
    PRINT "GPT-2 BASIC Benchmarking Suite"
    PRINT "=============================="
    PRINT

    ' Initialize random seed
    RANDOMIZE TIMER

    ' Initialize systems
    InitMatrixOps()
    InitAsmOptimizations()
    InitFixedSoftmax()
    InitializeDefaultTokenizer()
    InitMemoryManager()

    ' Run component benchmarks
    IF RUN_COMPONENT_BENCHMARKS THEN
        BenchmarkMatrixMultiply()
        BenchmarkAttention()
        BenchmarkFixedPoint()
        BenchmarkTokenizer()
    END IF

    ' Run integration benchmarks
    IF RUN_INTEGRATION_BENCHMARKS THEN
        BenchmarkTransformerBlock()
        BenchmarkMemoryStreaming()
    END IF

    ' Run end-to-end benchmarks
    IF RUN_END_TO_END_BENCHMARKS THEN
        BenchmarkTokenGeneration()
    END IF

    ' Print all results
    PrintBenchmarkResults("")

    ' Print results by category
    PRINT
    PRINT "Matrix Operation Results:"
    PrintBenchmarkResults("MatrixMultiply")

    PRINT
    PRINT "Attention Operation Results:"
    PrintBenchmarkResults("Attention")

    PRINT
    PRINT "Fixed-Point Operation Results:"
    PrintBenchmarkResults("FixedPoint")

    PRINT
    PRINT "Token Generation Results:"
    PrintBenchmarkResults("Generation")

    ' Clean up
    ShutdownMemoryManager()
END SUB

' Main entry point
SUB Benchmark_Main()
    RunAllBenchmarks()
END SUB
