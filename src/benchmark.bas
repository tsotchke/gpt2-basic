' Performance benchmarking for the GPT-2 in BASIC implementation.
' This file provides tools to measure and analyze the performance
' of various components of the system.

' Include necessary files
#INCLUDE "data_structures.bas"
#INCLUDE "quantization.bas"
#INCLUDE "matrix_ops.bas"
#INCLUDE "transformer_components.bas"
#INCLUDE "softmax_fixed.bas"
#INCLUDE "block_sparse.bas"
#INCLUDE "file_io.bas"
#INCLUDE "tokenizer.bas"
#INCLUDE "simd_ops.bas"
#INCLUDE "model.bas"

' Constants for benchmarking
CONST DEFAULT_MATRIX_SIZE AS INTEGER = 32
CONST DEFAULT_CONTEXT_LENGTH AS INTEGER = 64
CONST DEFAULT_VOCAB_SIZE AS INTEGER = 1000
CONST DEFAULT_EMBEDDING_DIM AS INTEGER = 128
CONST DEFAULT_BENCHMARK_ITERATIONS AS INTEGER = 5
CONST DEFAULT_WARMUP_ITERATIONS AS INTEGER = 2

' --- Data structure for benchmark results ---
TYPE BenchmarkResult
    name AS STRING * 64
    operation AS STRING * 64
    iterations AS INTEGER
    total_time AS DOUBLE
    avg_time AS DOUBLE
    min_time AS DOUBLE
    max_time AS DOUBLE
    throughput AS DOUBLE ' Operations per second
END TYPE

' Global array to store benchmark results
DIM shared_results(100) AS BenchmarkResult
DIM shared_result_count AS INTEGER

' --- Timing utilities ---
FUNCTION GetTimeStamp() AS DOUBLE
    ' Return current timestamp in seconds
    FUNCTION = TIMER
END FUNCTION

' Format seconds to a readable string (e.g., "1.234 s" or "123.4 ms")
FUNCTION FormatTime(seconds AS DOUBLE) AS STRING
    IF seconds >= 1.0 THEN
        FUNCTION = STR$(seconds) + " s"
    ELSEIF seconds >= 0.001 THEN
        FUNCTION = STR$(seconds * 1000) + " ms"
    ELSE
        FUNCTION = STR$(seconds * 1000000) + " µs"
    END IF
END FUNCTION

' --- Matrix benchmarking functions ---

' Benchmark matrix multiplication
SUB BenchmarkMatrixMultiply(size AS INTEGER, iterations AS INTEGER, use_simd AS INTEGER)
    DIM name AS STRING
    IF use_simd THEN
        name = "Matrix Multiply (SIMD)"
    ELSE
        name = "Matrix Multiply"
    END IF

    DIM a AS Matrix, b AS Matrix, c AS Matrix
    DIM times(iterations - 1) AS DOUBLE
    DIM total_time AS DOUBLE = 0
    DIM min_time AS DOUBLE = 1E+30
    DIM max_time AS DOUBLE = 0
    
    ' Initialize matrices with random data
    InitMatrix a, size, size
    InitMatrix b, size, size
    InitMatrix c, size, size
    
    DIM i AS INTEGER, j AS INTEGER, k AS INTEGER
    FOR i = 0 TO a.rows - 1
        FOR j = 0 TO a.cols - 1
            a.data(i, j) = INT(RND * 16)
        NEXT j
    NEXT i
    
    FOR i = 0 TO b.rows - 1
        FOR j = 0 TO b.cols - 1
            b.data(i, j) = INT(RND * 16)
        NEXT j
    NEXT i
    
    ' Warmup iterations (not counted in timing)
    FOR i = 1 TO DEFAULT_WARMUP_ITERATIONS
        IF use_simd THEN
            MatrixMultiply_SIMD a, b, c
        ELSE
            MatrixMultiply a, b, c
        END IF
    NEXT i
    
    ' Timed iterations
    FOR i = 0 TO iterations - 1
        DIM start_time AS DOUBLE = GetTimeStamp()
        
        IF use_simd THEN
            MatrixMultiply_SIMD a, b, c
        ELSE
            MatrixMultiply a, b, c
        END IF
        
        DIM end_time AS DOUBLE = GetTimeStamp()
        DIM elapsed AS DOUBLE = end_time - start_time
        
        times(i) = elapsed
        total_time = total_time + elapsed
        
        IF elapsed < min_time THEN min_time = elapsed
        IF elapsed > max_time THEN max_time = elapsed
    NEXT i
    
    ' Calculate average
    DIM avg_time AS DOUBLE = total_time / iterations
    
    ' Calculate operations: roughly 2*n³ for matrix multiply of size n×n
    DIM operations AS DOUBLE = 2 * size * size * size
    DIM throughput AS DOUBLE = operations / avg_time
    
    ' Store result
    shared_results(shared_result_count).name = name
    shared_results(shared_result_count).operation = "Size " + STR$(size) + "x" + STR$(size)
    shared_results(shared_result_count).iterations = iterations
    shared_results(shared_result_count).total_time = total_time
    shared_results(shared_result_count).avg_time = avg_time
    shared_results(shared_result_count).min_time = min_time
    shared_results(shared_result_count).max_time = max_time
    shared_results(shared_result_count).throughput = throughput
    
    shared_result_count = shared_result_count + 1
    
    ' Clean up
    FreeMatrix a
    FreeMatrix b
    FreeMatrix c
END SUB

' Benchmark attention mechanism
SUB BenchmarkAttention(ctx_length AS INTEGER, embed_dim AS INTEGER, num_heads AS INTEGER, iterations AS INTEGER, use_sparse AS INTEGER)
    DIM name AS STRING
    IF use_sparse THEN
        name = "Block-Sparse Attention"
    ELSE
        name = "Dense Attention"
    END IF

    DIM input AS Matrix, q AS Matrix, k AS Matrix, v AS Matrix, o AS Matrix
    DIM output AS Matrix
    DIM times(iterations - 1) AS DOUBLE
    DIM total_time AS DOUBLE = 0
    DIM min_time AS DOUBLE = 1E+30
    DIM max_time AS DOUBLE = 0
    
    ' Initialize matrices
    InitMatrix input, ctx_length, embed_dim
    InitMatrix q, embed_dim, embed_dim
    InitMatrix k, embed_dim, embed_dim
    InitMatrix v, embed_dim, embed_dim
    InitMatrix o, embed_dim, embed_dim
    InitMatrix output, ctx_length, embed_dim
    
    ' Fill with random data
    DIM i AS INTEGER, j AS INTEGER
    FOR i = 0 TO input.rows - 1
        FOR j = 0 TO input.cols - 1
            input.data(i, j) = INT(RND * 16)
        NEXT j
    NEXT i
    
    FOR i = 0 TO q.rows - 1
        FOR j = 0 TO q.cols - 1
            q.data(i, j) = INT(RND * 16)
            k.data(i, j) = INT(RND * 16)
            v.data(i, j) = INT(RND * 16)
            o.data(i, j) = INT(RND * 16)
        NEXT j
    NEXT i
    
    ' Warmup iterations
    FOR i = 1 TO DEFAULT_WARMUP_ITERATIONS
        MultiHeadAttention input, q, k, v, o, output
    NEXT i
    
    ' Timed iterations
    FOR i = 0 TO iterations - 1
        DIM start_time AS DOUBLE = GetTimeStamp()
        
        MultiHeadAttention input, q, k, v, o, output
        
        DIM end_time AS DOUBLE = GetTimeStamp()
        DIM elapsed AS DOUBLE = end_time - start_time
        
        times(i) = elapsed
        total_time = total_time + elapsed
        
        IF elapsed < min_time THEN min_time = elapsed
        IF elapsed > max_time THEN max_time = elapsed
    NEXT i
    
    ' Calculate average
    DIM avg_time AS DOUBLE = total_time / iterations
    
    ' Calculate operations (approximate)
    ' For each head: 3 matrix muls (QKV projection) + attention calculation + output projection
    DIM head_dim AS INTEGER = embed_dim \ num_heads
    DIM operations AS DOUBLE = num_heads * ( _
        3 * ctx_length * embed_dim * head_dim + _ ' QKV projections
        ctx_length * ctx_length * head_dim + _    ' Attention scores
        ctx_length * ctx_length * head_dim + _    ' Attention application
        ctx_length * head_dim * embed_dim)        ' Output projection
    DIM throughput AS DOUBLE = operations / avg_time
    
    ' Store result
    shared_results(shared_result_count).name = name
    shared_results(shared_result_count).operation = "Ctx " + STR$(ctx_length) + ", Dim " + STR$(embed_dim)
    shared_results(shared_result_count).iterations = iterations
    shared_results(shared_result_count).total_time = total_time
    shared_results(shared_result_count).avg_time = avg_time
    shared_results(shared_result_count).min_time = min_time
    shared_results(shared_result_count).max_time = max_time
    shared_results(shared_result_count).throughput = throughput
    
    shared_result_count = shared_result_count + 1
    
    ' Clean up
    FreeMatrix input
    FreeMatrix q
    FreeMatrix k
    FreeMatrix v
    FreeMatrix o
    FreeMatrix output
END SUB

' Benchmark tokenization
SUB BenchmarkTokenization(text_length AS INTEGER, iterations AS INTEGER)
    DIM text AS STRING
    DIM times(iterations - 1) AS DOUBLE
    DIM total_time AS DOUBLE = 0
    DIM min_time AS DOUBLE = 1E+30
    DIM max_time AS DOUBLE = 0
    
    ' Generate random text for tokenization
    text = GenerateRandomText(text_length)
    
    ' Initialize tokenizer
    InitTokenizer()
    
    ' Create sample vocabulary if needed
    IF NOT FileExists("model_data/vocabulary.txt") THEN
        CreateSimpleVocabulary "model_data/vocabulary.txt"
    END IF
    
    ' Load vocabulary
    LoadVocabulary "model_data/vocabulary.txt"
    
    ' Warmup iterations
    FOR i = 1 TO DEFAULT_WARMUP_ITERATIONS
        DIM tokens() AS INTEGER = Tokenize(text, DEFAULT_CONTEXT_LENGTH)
    NEXT i
    
    ' Timed iterations
    FOR i = 0 TO iterations - 1
        DIM start_time AS DOUBLE = GetTimeStamp()
        
        DIM tokens() AS INTEGER = Tokenize(text, DEFAULT_CONTEXT_LENGTH)
        
        DIM end_time AS DOUBLE = GetTimeStamp()
        DIM elapsed AS DOUBLE = end_time - start_time
        
        times(i) = elapsed
        total_time = total_time + elapsed
        
        IF elapsed < min_time THEN min_time = elapsed
        IF elapsed > max_time THEN max_time = elapsed
    NEXT i
    
    ' Calculate average
    DIM avg_time AS DOUBLE = total_time / iterations
    
    ' Calculate throughput (characters per second)
    DIM throughput AS DOUBLE = text_length / avg_time
    
    ' Store result
    shared_results(shared_result_count).name = "Tokenization"
    shared_results(shared_result_count).operation = "Length " + STR$(text_length)
    shared_results(shared_result_count).iterations = iterations
    shared_results(shared_result_count).total_time = total_time
    shared_results(shared_result_count).avg_time = avg_time
    shared_results(shared_result_count).min_time = min_time
    shared_results(shared_result_count).max_time = max_time
    shared_results(shared_result_count).throughput = throughput
    
    shared_result_count = shared_result_count + 1
END SUB

' Benchmark Softmax implementation
SUB BenchmarkSoftmax(vector_size AS INTEGER, iterations AS INTEGER)
    DIM logits AS Matrix
    DIM times(iterations - 1) AS DOUBLE
    DIM total_time AS DOUBLE = 0
    DIM min_time AS DOUBLE = 1E+30
    DIM max_time AS DOUBLE = 0
    
    ' Initialize matrix
    InitMatrix logits, 1, vector_size
    
    ' Fill with random data
    DIM j AS INTEGER
    FOR j = 0 TO vector_size - 1
        ' Use scaled random values to simulate logits
        logits.data(0, j) = FixedToLogQuantized(FloatToFixed((RND - 0.5) * 10)).packed_value
    NEXT j
    
    ' Make sure lookup tables are initialized
    InitExpLookupTable()
    
    ' Warmup iterations
    FOR i = 1 TO DEFAULT_WARMUP_ITERATIONS
        SoftmaxVectorFixedPoint logits
    NEXT i
    
    ' Timed iterations
    FOR i = 0 TO iterations - 1
        ' Re-copy the original data for each iteration
        FOR j = 0 TO vector_size - 1
            logits.data(0, j) = FixedToLogQuantized(FloatToFixed((RND - 0.5) * 10)).packed_value
        NEXT j
        
        DIM start_time AS DOUBLE = GetTimeStamp()
        
        SoftmaxVectorFixedPoint logits
        
        DIM end_time AS DOUBLE = GetTimeStamp()
        DIM elapsed AS DOUBLE = end_time - start_time
        
        times(i) = elapsed
        total_time = total_time + elapsed
        
        IF elapsed < min_time THEN min_time = elapsed
        IF elapsed > max_time THEN max_time = elapsed
    NEXT i
    
    ' Calculate average
    DIM avg_time AS DOUBLE = total_time / iterations
    
    ' Calculate throughput (elements per second)
    DIM throughput AS DOUBLE = vector_size / avg_time
    
    ' Store result
    shared_results(shared_result_count).name = "Fixed-Point Softmax"
    shared_results(shared_result_count).operation = "Size " + STR$(vector_size)
    shared_results(shared_result_count).iterations = iterations
    shared_results(shared_result_count).total_time = total_time
    shared_results(shared_result_count).avg_time = avg_time
    shared_results(shared_result_count).min_time = min_time
    shared_results(shared_result_count).max_time = max_time
    shared_results(shared_result_count).throughput = throughput
    
    shared_result_count = shared_result_count + 1
    
    ' Clean up
    FreeMatrix logits
END SUB

' Benchmark forward pass
SUB BenchmarkForwardPass(ctx_length AS INTEGER, embed_dim AS INTEGER, num_layers AS INTEGER, iterations AS INTEGER)
    ' This is a simplified benchmark focusing just on the core forward pass without full model loading
    
    DIM tokens(ctx_length - 1) AS INTEGER
    DIM logits AS Matrix
    DIM times(iterations - 1) AS DOUBLE
    DIM total_time AS DOUBLE = 0
    DIM min_time AS DOUBLE = 1E+30
    DIM max_time AS DOUBLE = 0
    
    ' Initialize with random token IDs
    DIM i AS INTEGER
    FOR i = 0 TO ctx_length - 1
        tokens(i) = INT(RND * 100) ' Random token IDs
    NEXT i
    
    ' Initialize logits matrix
    InitMatrix logits, ctx_length, DEFAULT_VOCAB_SIZE
    
    ' Set up a minimal model environment
    ' Note: This doesn't load actual weights or perform a complete forward pass,
    ' it just measures the operation of ForwardPass given the structure
    
    ' Initialize proper global variables in model.bas
    EMBEDDING_DIM = embed_dim
    NUM_LAYERS = num_layers
    CONTEXT_LENGTH = ctx_length
    VOCAB_SIZE = DEFAULT_VOCAB_SIZE
    
    ' Load model in a test mode
    ' We'll use non-streaming mode with random weights
    LoadModelParameters "model_weights.bin" ' This will initialize with placeholder values
    
    ' Warmup iterations
    FOR i = 1 TO DEFAULT_WARMUP_ITERATIONS
        ForwardPass tokens, logits
    NEXT i
    
    ' Timed iterations
    FOR i = 0 TO iterations - 1
        DIM start_time AS DOUBLE = GetTimeStamp()
        
        ForwardPass tokens, logits
        
        DIM end_time AS DOUBLE = GetTimeStamp()
        DIM elapsed AS DOUBLE = end_time - start_time
        
        times(i) = elapsed
        total_time = total_time + elapsed
        
        IF elapsed < min_time THEN min_time = elapsed
        IF elapsed > max_time THEN max_time = elapsed
    NEXT i
    
    ' Calculate average
    DIM avg_time AS DOUBLE = total_time / iterations
    
    ' Calculate throughput (tokens per second for a full pass)
    DIM throughput AS DOUBLE = ctx_length / avg_time
    
    ' Store result
    shared_results(shared_result_count).name = "Forward Pass"
    shared_results(shared_result_count).operation = "Ctx " + STR$(ctx_length) + ", Layers " + STR$(num_layers)
    shared_results(shared_result_count).iterations = iterations
    shared_results(shared_result_count).total_time = total_time
    shared_results(shared_result_count).avg_time = avg_time
    shared_results(shared_result_count).min_time = min_time
    shared_results(shared_result_count).max_time = max_time
    shared_results(shared_result_count).throughput = throughput
    
    shared_result_count = shared_result_count + 1
    
    ' Clean up
    FreeMatrix logits
    FreeModelResources()
END SUB

' --- Utility functions ---

' Generate random text of specified length
FUNCTION GenerateRandomText(length AS INTEGER) AS STRING
    DIM result AS STRING
    DIM chars AS STRING = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!? "
    DIM i AS INTEGER
    
    FOR i = 1 TO length
        DIM char_idx AS INTEGER = INT(RND * LEN(chars)) + 1
        result = result + MID$(chars, char_idx, 1)
    NEXT i
    
    FUNCTION = result
END FUNCTION

' Check if a file exists
FUNCTION FileExists(filepath AS STRING) AS INTEGER
    DIM file_num AS INTEGER
    
    file_num = FREEFILE
    
    ON ERROR GOTO FileError
    OPEN filepath FOR INPUT AS #file_num
    CLOSE #file_num
    ON ERROR GOTO 0
    
    ' If we get here, the file exists
    FUNCTION = 1
    EXIT FUNCTION
    
FileError:
    FUNCTION = 0
END FUNCTION

' --- Results display functions ---

' Print benchmark results in a table format
SUB PrintResults()
    PRINT "======================================================================"
    PRINT "BENCHMARK RESULTS"
    PRINT "======================================================================"
    PRINT "Operation                 | Parameters         | Avg Time    | Throughput"
    PRINT "----------------------------------------------------------------------"
    
    DIM i AS INTEGER
    FOR i = 0 TO shared_result_count - 1
        DIM name AS STRING = RTRIM$(shared_results(i).name)
        DIM operation AS STRING = RTRIM$(shared_results(i).operation)
        DIM avg_time AS STRING = FormatTime(shared_results(i).avg_time)
        DIM throughput AS STRING
        
        ' Format throughput based on operation type
        IF INSTR(name, "Matrix Multiply") > 0 THEN
            ' For matrix ops, show GFLOPS (approximate)
            throughput = STR$(shared_results(i).throughput / 1E9) + " GFLOPS"
        ELSEIF INSTR(name, "Attention") > 0 THEN
            ' For attention, show operations per second
            throughput = STR$(shared_results(i).throughput / 1E9) + " GOPS"
        ELSEIF INSTR(name, "Tokenization") > 0 THEN
            ' For tokenization, show chars per second
            throughput = STR$(shared_results(i).throughput) + " chars/s"
        ELSEIF INSTR(name, "Softmax") > 0 THEN
            ' For softmax, show elements per second
            throughput = STR$(shared_results(i).throughput) + " elem/s"
        ELSEIF INSTR(name, "Forward Pass") > 0 THEN
            ' For forward pass, show tokens per second
            throughput = STR$(shared_results(i).throughput) + " tokens/s"
        ELSE
            ' Default throughput display
            throughput = STR$(shared_results(i).throughput) + " ops/s"
        END IF
        
        ' Format output columns for nice alignment
        PRINT LEFT$(name + SPACE$(25), 25); " | ";
        PRINT LEFT$(operation + SPACE$(20), 20); " | ";
        PRINT LEFT$(avg_time + SPACE$(12), 12); " | ";
        PRINT throughput
    NEXT i
    
    PRINT "======================================================================"
END SUB

' Save benchmark results to a file
SUB SaveResultsToFile(filepath AS STRING)
    DIM file_num AS INTEGER
    
    file_num = FREEFILE
    
    ON ERROR GOTO SaveError
    OPEN filepath FOR OUTPUT AS #file_num
    ON ERROR GOTO 0
    
    ' Write header
    PRINT #file_num, "Operation,Parameters,Iterations,Total Time,Avg Time,Min Time,Max Time,Throughput"
    
    ' Write data
    DIM i AS INTEGER
    FOR i = 0 TO shared_result_count - 1
        PRINT #file_num, _
            RTRIM$(shared_results(i).name); ","; _
            RTRIM$(shared_results(i).operation); ","; _
            shared_results(i).iterations; ","; _
            shared_results(i).total_time; ","; _
            shared_results(i).avg_time; ","; _
            shared_results(i).min_time; ","; _
            shared_results(i).max_time; ","; _
            shared_results(i).throughput
    NEXT i
    
    CLOSE #file_num
    PRINT "Results saved to "; filepath
    EXIT SUB
    
SaveError:
    PRINT "Error saving results to "; filepath
    CLOSE #file_num
END SUB

' --- Main benchmark function ---

SUB RunBenchmarks()
    DIM i AS INTEGER
    
    ' Initialize
    shared_result_count = 0
    RANDOMIZE TIMER
    
    PRINT "Starting GPT-2 in BASIC benchmarks..."
    PRINT "This will take some time. Please wait..."
    PRINT
    
    ' Matrix multiplication benchmarks (standard and SIMD)
    FOR i = 1 TO 3
        DIM size AS INTEGER = 8 * (2 ^ i) ' 16, 32, 64
        BenchmarkMatrixMultiply size, DEFAULT_BENCHMARK_ITERATIONS, 0 ' Standard
        BenchmarkMatrixMultiply size, DEFAULT_BENCHMARK_ITERATIONS, 1 ' SIMD
    NEXT i
    
    ' Attention mechanism benchmarks
    FOR i = 1 TO 2
        DIM ctx_len AS INTEGER = 32 * i ' 32, 64
        BenchmarkAttention ctx_len, DEFAULT_EMBEDDING_DIM, 4, DEFAULT_BENCHMARK_ITERATIONS, 0 ' Dense
        BenchmarkAttention ctx_len, DEFAULT_EMBEDDING_DIM, 4, DEFAULT_BENCHMARK_ITERATIONS, 1 ' Sparse
    NEXT i
    
    ' Tokenization benchmarks
    BenchmarkTokenization 100, DEFAULT_BENCHMARK_ITERATIONS
    BenchmarkTokenization 500, DEFAULT_BENCHMARK_ITERATIONS
    BenchmarkTokenization 1000, DEFAULT_BENCHMARK_ITERATIONS
    
    ' Softmax benchmarks
    BenchmarkSoftmax DEFAULT_VOCAB_SIZE / 4, DEFAULT_BENCHMARK_ITERATIONS
    BenchmarkSoftmax DEFAULT_VOCAB_SIZE, DEFAULT_BENCHMARK_ITERATIONS
    
    ' Forward pass benchmarks
    BenchmarkForwardPass 32, DEFAULT_EMBEDDING_DIM, 2, 2 ' Smaller context, fewer layers, fewer iterations
    BenchmarkForwardPass 64, DEFAULT_EMBEDDING_DIM, 2, 2
    
    ' Print and save results
    PrintResults()
    SaveResultsToFile "benchmark_results.csv"
END SUB

' Main entry point
PRINT "GPT-2 in BASIC Benchmark Suite"
PRINT "-----------------------------"
PRINT

RunBenchmarks()

END
