# Memory Management Design Document

This document details the design and implementation of the memory management system for the GPT-2 BASIC project, ensuring efficient operation within 486-era hardware constraints.

## Overview

Memory management is critical for implementing transformer models on 486-era hardware with limited RAM (typically 4-32MB). Our system provides comprehensive memory tracking, allocation optimization, matrix pooling, and parameter streaming to minimize memory usage while maintaining performance.

## Design Goals

1. **Memory Constraint Compliance**: Ensure operation within 32MB RAM limit
2. **Efficient Allocation**: Minimize fragmentation and overhead
3. **Resource Tracking**: Provide visibility into memory usage patterns
4. **Matrix Reuse**: Enable pooling and recycling of matrix memory
5. **Streaming Support**: Facilitate loading parameters from disk as needed
6. **Graceful Degradation**: Adapt to limited memory environments

## Core Components

### Memory Tracker

```basic
' Global memory tracking structure
TYPE MemoryTracker
    total_allocated AS LONG     ' Total memory currently allocated (bytes)
    peak_allocated AS LONG      ' Peak memory usage (bytes)
    num_allocations AS INTEGER  ' Number of active allocations
    memory_limit AS LONG        ' Maximum allowed memory usage (bytes)
    allocation_map AS LONG PTR  ' Optional pointer to allocation tracking array
    track_detailed AS INTEGER   ' Whether detailed tracking is enabled
END TYPE

DIM SHARED g_memory_tracker AS MemoryTracker

' Memory allocation record for detailed tracking
TYPE AllocationRecord
    ptr AS ANY PTR      ' Pointer to allocated memory
    size AS LONG        ' Size of allocation in bytes
    tag AS STRING * 32  ' Optional tag to identify allocation purpose
    line_num AS INTEGER ' Source line number
    file_id AS INTEGER  ' Source file identifier
    is_active AS INTEGER ' Whether this record is active
END TYPE
```

### Memory Management Functions

```basic
' Initialize the memory tracker
SUB InitMemoryTracker(memory_limit AS LONG, detailed_tracking AS INTEGER)
    g_memory_tracker.total_allocated = 0
    g_memory_tracker.peak_allocated = 0
    g_memory_tracker.num_allocations = 0
    g_memory_tracker.memory_limit = memory_limit
    g_memory_tracker.track_detailed = detailed_tracking
    
    ' Initialize detailed tracking if requested
    IF detailed_tracking THEN
        g_memory_tracker.allocation_map = ALLOCATE(SIZEOF(AllocationRecord) * 1024)
        DIM i AS INTEGER
        FOR i = 0 TO 1023
            DIM record AS AllocationRecord PTR = g_memory_tracker.allocation_map + SIZEOF(AllocationRecord) * i
            record->is_active = FALSE
        NEXT i
    END IF
END SUB

' Clean up the memory tracker
SUB CleanupMemoryTracker()
    IF g_memory_tracker.track_detailed THEN
        DEALLOCATE(g_memory_tracker.allocation_map)
    END IF
    
    ' Verify all memory has been freed
    IF g_memory_tracker.total_allocated > 0 THEN
        PRINT "WARNING: Memory leak detected! "; g_memory_tracker.total_allocated; " bytes still allocated."
    END IF
END SUB

' Track a memory allocation
FUNCTION TrackedAllocate(size AS LONG, tag AS STRING, line_num AS INTEGER, file_id AS INTEGER) AS ANY PTR
    ' Check if allocation would exceed memory limit
    IF g_memory_tracker.total_allocated + size > g_memory_tracker.memory_limit THEN
        PRINT "ERROR: Memory allocation of "; size; " bytes would exceed memory limit of "; g_memory_tracker.memory_limit; " bytes"
        PRINT "       Current usage: "; g_memory_tracker.total_allocated; " bytes"
        RETURN NULL
    END IF
    
    ' Allocate memory
    DIM ptr AS ANY PTR = ALLOCATE(size)
    IF ptr = NULL THEN
        PRINT "ERROR: Failed to allocate "; size; " bytes"
        RETURN NULL
    END IF
    
    ' Update tracking information
    g_memory_tracker.total_allocated = g_memory_tracker.total_allocated + size
    g_memory_tracker.num_allocations = g_memory_tracker.num_allocations + 1
    
    ' Update peak if needed
    IF g_memory_tracker.total_allocated > g_memory_tracker.peak_allocated THEN
        g_memory_tracker.peak_allocated = g_memory_tracker.total_allocated
    END IF
    
    ' Record detailed allocation information if enabled
    IF g_memory_tracker.track_detailed THEN
        DIM i AS INTEGER
        FOR i = 0 TO 1023
            DIM record AS AllocationRecord PTR = g_memory_tracker.allocation_map + SIZEOF(AllocationRecord) * i
            IF NOT record->is_active THEN
                record->ptr = ptr
                record->size = size
                record->tag = LEFT$(tag, 32)
                record->line_num = line_num
                record->file_id = file_id
                record->is_active = TRUE
                EXIT FOR
            END IF
        NEXT i
    END IF
    
    RETURN ptr
END FUNCTION

' Track a memory deallocation
SUB TrackedDeallocate(ptr AS ANY PTR)
    IF ptr = NULL THEN
        PRINT "WARNING: Attempting to deallocate NULL pointer"
        RETURN
    END IF
    
    ' Find allocation record if detailed tracking is enabled
    IF g_memory_tracker.track_detailed THEN
        DIM i AS INTEGER
        DIM size AS LONG = 0
        FOR i = 0 TO 1023
            DIM record AS AllocationRecord PTR = g_memory_tracker.allocation_map + SIZEOF(AllocationRecord) * i
            IF record->is_active AND record->ptr = ptr THEN
                size = record->size
                record->is_active = FALSE
                EXIT FOR
            END IF
        NEXT i
        
        IF size = 0 THEN
            PRINT "ERROR: Attempting to deallocate untracked pointer"
            RETURN
        END IF
        
        ' Update tracking information
        g_memory_tracker.total_allocated = g_memory_tracker.total_allocated - size
        g_memory_tracker.num_allocations = g_memory_tracker.num_allocations - 1
    END IF
    
    ' Actually deallocate the memory
    DEALLOCATE(ptr)
END SUB

' Get memory usage report
SUB GetMemoryUsageReport(BYREF report AS STRING)
    DIM usage_pct AS SINGLE = (g_memory_tracker.total_allocated * 100.0) / g_memory_tracker.memory_limit
    
    report = "Memory Usage Report:" + CHR$(10)
    report = report + "Current: " + STR$(g_memory_tracker.total_allocated) + " bytes (" + STR$(usage_pct) + "% of limit)" + CHR$(10)
    report = report + "Peak: " + STR$(g_memory_tracker.peak_allocated) + " bytes" + CHR$(10)
    report = report + "Allocations: " + STR$(g_memory_tracker.num_allocations) + CHR$(10)
    report = report + "Limit: " + STR$(g_memory_tracker.memory_limit) + " bytes" + CHR$(10)
    
    ' Add detailed allocation breakdown if tracking is enabled
    IF g_memory_tracker.track_detailed THEN
        report = report + CHR$(10) + "Top allocations by size:" + CHR$(10)
        
        ' Find and sort the top allocations
        DIM allocation_sizes(0 TO 9) AS LONG
        DIM allocation_tags(0 TO 9) AS STRING * 32
        DIM i AS INTEGER, j AS INTEGER
        
        FOR i = 0 TO 1023
            DIM record AS AllocationRecord PTR = g_memory_tracker.allocation_map + SIZEOF(AllocationRecord) * i
            IF record->is_active THEN
                ' Check if this allocation should be in the top 10
                FOR j = 0 TO 9
                    IF record->size > allocation_sizes(j) THEN
                        ' Shift everything down
                        DIM k AS INTEGER
                        FOR k = 9 TO j + 1 STEP -1
                            allocation_sizes(k) = allocation_sizes(k - 1)
                            allocation_tags(k) = allocation_tags(k - 1)
                        NEXT k
                        
                        ' Insert this allocation
                        allocation_sizes(j) = record->size
                        allocation_tags(j) = record->tag
                        EXIT FOR
                    END IF
                NEXT j
            END IF
        NEXT i
        
        ' Add the top allocations to the report
        FOR i = 0 TO 9
            IF allocation_sizes(i) > 0 THEN
                report = report + RTRIM$(allocation_tags(i)) + ": " + STR$(allocation_sizes(i)) + " bytes" + CHR$(10)
            END IF
        NEXT i
    END IF
END SUB
```

### Matrix Memory Pool

The matrix memory pool allows reuse of allocated matrices, reducing allocation/deallocation overhead and memory fragmentation.

```basic
' Matrix pool structure for efficient matrix memory reuse
TYPE MatrixPool
    pool_entries AS INTEGER    ' Number of entries in the pool
    matrices() AS Matrix       ' Array of pooled matrices
    in_use() AS INTEGER        ' Whether each matrix is currently in use
    total_bytes AS LONG        ' Total memory used by the pool
END TYPE

DIM SHARED g_matrix_pool AS MatrixPool

' Initialize the matrix pool
SUB InitMatrixPool(pool_size AS INTEGER)
    g_matrix_pool.pool_entries = pool_size
    REDIM g_matrix_pool.matrices(1 TO pool_size)
    REDIM g_matrix_pool.in_use(1 TO pool_size)
    
    ' Initialize all matrices as not in use
    DIM i AS INTEGER
    FOR i = 1 TO pool_size
        g_matrix_pool.in_use(i) = FALSE
    NEXT i
    
    g_matrix_pool.total_bytes = 0
END SUB

' Get a matrix from the pool, or allocate a new one if necessary
FUNCTION GetMatrixFromPool(rows AS INTEGER, cols AS INTEGER) AS INTEGER
    DIM i AS INTEGER
    DIM entry_idx AS INTEGER = 0
    
    ' First, look for an existing matrix of the right size
    FOR i = 1 TO g_matrix_pool.pool_entries
        IF NOT g_matrix_pool.in_use(i) AND _
           g_matrix_pool.matrices(i).rows = rows AND _
           g_matrix_pool.matrices(i).cols = cols THEN
            entry_idx = i
            EXIT FOR
        END IF
    NEXT i
    
    ' If no exact match, find a matrix of the right size or larger
    IF entry_idx = 0 THEN
        DIM best_fit_idx AS INTEGER = 0
        DIM best_fit_size AS LONG = 0
        
        FOR i = 1 TO g_matrix_pool.pool_entries
            IF NOT g_matrix_pool.in_use(i) AND _
               g_matrix_pool.matrices(i).rows >= rows AND _
               g_matrix_pool.matrices(i).cols >= cols THEN
                DIM size AS LONG = g_matrix_pool.matrices(i).rows * g_matrix_pool.matrices(i).cols
                IF best_fit_idx = 0 OR size < best_fit_size THEN
                    best_fit_idx = i
                    best_fit_size = size
                END IF
            END IF
        NEXT i
        
        entry_idx = best_fit_idx
    END IF
    
    ' If still no match, find an unused entry
    IF entry_idx = 0 THEN
        FOR i = 1 TO g_matrix_pool.pool_entries
            IF NOT g_matrix_pool.in_use(i) THEN
                entry_idx = i
                EXIT FOR
            END IF
        NEXT i
    END IF
    
    ' If we found a usable entry, initialize and return it
    IF entry_idx > 0 THEN
        ' If we're reusing a matrix of different size, free its memory first
        IF g_matrix_pool.matrices(entry_idx).data <> NULL AND _
           (g_matrix_pool.matrices(entry_idx).rows <> rows OR _
            g_matrix_pool.matrices(entry_idx).cols <> cols) THEN
            DIM old_size AS LONG = g_matrix_pool.matrices(entry_idx).rows * g_matrix_pool.matrices(entry_idx).cols * 4
            g_matrix_pool.total_bytes = g_matrix_pool.total_bytes - old_size
            DEALLOCATE(g_matrix_pool.matrices(entry_idx).data)
            g_matrix_pool.matrices(entry_idx).data = NULL
        END IF
        
        ' Initialize the matrix if needed
        IF g_matrix_pool.matrices(entry_idx).data = NULL THEN
            g_matrix_pool.matrices(entry_idx).rows = rows
            g_matrix_pool.matrices(entry_idx).cols = cols
            g_matrix_pool.matrices(entry_idx).data = TrackedAllocate(rows * cols * 4, "MatrixPool", 0, 0)
            g_matrix_pool.total_bytes = g_matrix_pool.total_bytes + (rows * cols * 4)
        END IF
        
        ' Mark as in use
        g_matrix_pool.in_use(entry_idx) = TRUE
        
        RETURN entry_idx
    END IF
    
    ' No available entries in the pool
    RETURN 0
END FUNCTION

' Return a matrix to the pool
SUB ReturnMatrixToPool(pool_idx AS INTEGER)
    IF pool_idx <= 0 OR pool_idx > g_matrix_pool.pool_entries THEN
        PRINT "ERROR: Invalid matrix pool index: "; pool_idx
        RETURN
    END IF
    
    ' Mark as not in use
    g_matrix_pool.in_use(pool_idx) = FALSE
END SUB

' Clean up the matrix pool
SUB CleanupMatrixPool()
    DIM i AS INTEGER
    
    FOR i = 1 TO g_matrix_pool.pool_entries
        IF g_matrix_pool.matrices(i).data <> NULL THEN
            TrackedDeallocate(g_matrix_pool.matrices(i).data)
            g_matrix_pool.matrices(i).data = NULL
        END IF
    NEXT i
    
    g_matrix_pool.total_bytes = 0
END SUB
```

### Parameter Streaming System

The parameter streaming system enables loading model parameters from disk on demand to minimize memory usage.

```basic
' Type definitions for model file format
TYPE ModelHeader
    file_signature AS STRING * 8   ' "GPT2BSCM" (GPT-2 BASIC Model)
    version AS INTEGER             ' File format version
    num_layers AS INTEGER          ' Number of transformer layers
    embed_dim AS INTEGER           ' Embedding dimension
    vocab_size AS INTEGER          ' Size of vocabulary
    num_heads AS INTEGER           ' Number of attention heads
    layer_offsets(0 TO 127) AS LONG ' File offsets for each layer
    embed_offset AS LONG           ' File offset for embedding layer
    output_offset AS LONG          ' File offset for output layer
END TYPE

TYPE LayerOffsets
    attention_qkv_offset AS LONG   ' Offset for attention QKV weights
    attention_out_offset AS LONG   ' Offset for attention output weights
    ffn_in_offset AS LONG          ' Offset for FFN input weights
    ffn_out_offset AS LONG         ' Offset for FFN output weights
    norm1_offset AS LONG           ' Offset for first layer norm
    norm2_offset AS LONG           ' Offset for second layer norm
END TYPE

' Stream handle for model parameters
TYPE ModelStream
    file_handle AS INTEGER         ' File handle
    file_path AS STRING            ' Path to model file
    header AS ModelHeader          ' Cached header information
    layer_offsets(0 TO 31) AS LayerOffsets ' Cached layer offset information
    current_layer AS INTEGER       ' Currently loaded layer (-1 if none)
    current_section AS INTEGER     ' Currently loaded section
    is_open AS INTEGER             ' Whether the stream is open
END TYPE

DIM SHARED g_model_stream AS ModelStream

' Open a model parameters file for streaming
FUNCTION OpenModelStream(file_path AS STRING) AS INTEGER
    ' Close any existing stream
    IF g_model_stream.is_open THEN
        CLOSE #g_model_stream.file_handle
    END IF
    
    ' Open the model file
    g_model_stream.file_handle = FREEFILE
    g_model_stream.file_path = file_path
    
    ON ERROR GOTO open_error
    OPEN file_path FOR BINARY ACCESS READ AS #g_model_stream.file_handle
    ON ERROR GOTO 0
    
    ' Read the header
    GET #g_model_stream.file_handle, 1, g_model_stream.header
    
    ' Verify file signature
    IF g_model_stream.header.file_signature <> "GPT2BSCM" THEN
        PRINT "ERROR: Invalid model file format"
        CLOSE #g_model_stream.file_handle
        RETURN FALSE
    END IF
    
    ' Read layer offsets
    DIM i AS INTEGER
    FOR i = 0 TO g_model_stream.header.num_layers - 1
        SEEK #g_model_stream.file_handle, g_model_stream.header.layer_offsets(i)
        GET #g_model_stream.file_handle, , g_model_stream.layer_offsets(i)
    NEXT i
    
    g_model_stream.current_layer = -1
    g_model_stream.current_section = -1
    g_model_stream.is_open = TRUE
    
    RETURN TRUE
    
open_error:
    PRINT "ERROR: Failed to open model file: "; file_path
    RETURN FALSE
END FUNCTION

' Load a specific layer's parameters
FUNCTION LoadLayerParameters(layer_idx AS INTEGER, BYREF attention_qkv AS Matrix, BYREF attention_out AS Matrix, _
                            BYREF ffn_in AS Matrix, BYREF ffn_out AS Matrix, _
                            BYREF norm1_scale AS Matrix, BYREF norm1_bias AS Matrix, _
                            BYREF norm2_scale AS Matrix, BYREF norm2_bias AS Matrix) AS INTEGER
    IF NOT g_model_stream.is_open THEN
        PRINT "ERROR: Model stream not open"
        RETURN FALSE
    END IF
    
    IF layer_idx < 0 OR layer_idx >= g_model_stream.header.num_layers THEN
        PRINT "ERROR: Invalid layer index: "; layer_idx
        RETURN FALSE
    END IF
    
    ' Load attention QKV weights
    SEEK #g_model_stream.file_handle, g_model_stream.layer_offsets(layer_idx).attention_qkv_offset
    LoadMatrixFromStream(g_model_stream.file_handle, attention_qkv)
    
    ' Load attention output weights
    SEEK #g_model_stream.file_handle, g_model_stream.layer_offsets(layer_idx).attention_out_offset
    LoadMatrixFromStream(g_model_stream.file_handle, attention_out)
    
    ' Load FFN input weights
    SEEK #g_model_stream.file_handle, g_model_stream.layer_offsets(layer_idx).ffn_in_offset
    LoadMatrixFromStream(g_model_stream.file_handle, ffn_in)
    
    ' Load FFN output weights
    SEEK #g_model_stream.file_handle, g_model_stream.layer_offsets(layer_idx).ffn_out_offset
    LoadMatrixFromStream(g_model_stream.file_handle, ffn_out)
    
    ' Load norm1 parameters
    SEEK #g_model_stream.file_handle, g_model_stream.layer_offsets(layer_idx).norm1_offset
    LoadMatrixFromStream(g_model_stream.file_handle, norm1_scale)
    LoadMatrixFromStream(g_model_stream.file_handle, norm1_bias)
    
    ' Load norm2 parameters
    SEEK #g_model_stream.file_handle, g_model_stream.layer_offsets(layer_idx).norm2_offset
    LoadMatrixFromStream(g_model_stream.file_handle, norm2_scale)
    LoadMatrixFromStream(g_model_stream.file_handle, norm2_bias)
    
    g_model_stream.current_layer = layer_idx
    
    RETURN TRUE
END FUNCTION

' Load a matrix from the stream
SUB LoadMatrixFromStream(file_handle AS INTEGER, BYREF matrix AS Matrix)
    DIM rows AS INTEGER, cols AS INTEGER
    
    ' Read dimensions
    GET #file_handle, , rows
    GET #file_handle, , cols
    
    ' Initialize matrix
    InitMatrix(matrix, rows, cols)
    
    ' Read data
    GET #file_handle, , matrix.data(0, 0), rows * cols
END SUB

' Close the model stream
SUB CloseModelStream()
    IF g_model_stream.is_open THEN
        CLOSE #g_model_stream.file_handle
        g_model_stream.is_open = FALSE
    END IF
END SUB
```

### Memory-Aware Matrix Operations

These functions adapt matrix operations based on available memory.

```basic
' Matrix multiplication with memory-aware chunking
SUB MemoryAwareMatrixMultiply(A AS Matrix, B AS Matrix, BYREF C AS Matrix)
    DIM max_chunk_size AS INTEGER
    DIM i AS INTEGER, j AS INTEGER, chunk AS INTEGER
    DIM chunk_start AS INTEGER, chunk_end AS INTEGER
    DIM temp_result AS Matrix
    
    ' Initialize result matrix
    InitMatrix(C, A.rows, B.cols)
    
    ' Determine maximum chunk size based on available memory
    DIM available_memory AS LONG = g_memory_tracker.memory_limit - g_memory_tracker.total_allocated
    max_chunk_size = MIN(A.rows, SQRT(available_memory / (8 * B.cols))) ' 8 bytes per entry (temp results)
    
    ' If we can do it in one go, use standard multiplication
    IF max_chunk_size >= A.rows THEN
        StandardMatrixMultiply(A, B, C)
        RETURN
    END IF
    
    ' Process in chunks
    FOR chunk_start = 0 TO A.rows - 1 STEP max_chunk_size
        chunk_end = MIN(chunk_start + max_chunk_size - 1, A.rows - 1)
        
        ' Extract chunk from matrix A
        DIM A_chunk AS Matrix
        InitMatrix(A_chunk, chunk_end - chunk_start + 1, A.cols)
        
        FOR i = chunk_start TO chunk_end
            FOR j = 0 TO A.cols - 1
                A_chunk.data(i - chunk_start, j) = A.data(i, j)
            NEXT j
        NEXT i
        
        ' Multiply chunk
        InitMatrix(temp_result, A_chunk.rows, B.cols)
        StandardMatrixMultiply(A_chunk, B, temp_result)
        
        ' Copy result to output matrix
        FOR i = 0 TO temp_result.rows - 1
            FOR j = 0 TO temp_result.cols - 1
                C.data(chunk_start + i, j) = temp_result.data(i, j)
            NEXT j
        NEXT i
        
        ' Clean up
        FreeMatrix(A_chunk)
        FreeMatrix(temp_result)
    NEXT chunk_start
END SUB

' Memory-aware attention mechanism
SUB MemoryAwareAttention(Q AS Matrix, K AS Matrix, V AS Matrix, BYREF output AS Matrix, mask_type AS INTEGER)
    DIM available_memory AS LONG = g_memory_tracker.memory_limit - g_memory_tracker.total_allocated
    DIM seq_len AS INTEGER = Q.rows
    DIM head_dim AS INTEGER = Q.cols
    
    ' Estimate memory needed for full attention
    DIM attention_memory AS LONG = seq_len * seq_len * 4 ' 4 bytes per float for attention weights
    
    ' If we have enough memory, use standard or sparse attention based on sequence length
    IF attention_memory <= available_memory * 0.7 THEN ' Leave 30% buffer
        IF ShouldUseSparseAttention(seq_len, head_dim, mask_type) THEN
            BlockSparseAttention(Q, K, V, output, mask_type)
        ELSE
            StandardAttention(Q, K, V, output, mask_type)
        END IF
    ELSE
        ' Not enough memory for full attention - use chunked approach
        ChunkedAttention(Q, K, V, output, SQRT(available_memory * 0.6 / 4), mask_type)
    END IF
END SUB

' Chunked attention for memory-constrained environments
SUB ChunkedAttention(Q AS Matrix, K AS Matrix, V AS Matrix, BYREF output AS Matrix, chunk_size AS INTEGER, mask_type AS INTEGER)
    DIM seq_len AS INTEGER = Q.rows
    DIM head_dim AS INTEGER = Q.cols
    DIM i AS INTEGER, j AS INTEGER
    DIM chunk_start AS INTEGER, chunk_end AS INTEGER
    DIM q_chunk AS Matrix, output_chunk AS Matrix
    
    ' Initialize output matrix
    InitMatrix(output, seq_len, head_dim)
    
    ' Process in chunks
    FOR chunk_start = 0 TO seq_len - 1 STEP chunk_size
        chunk_end = MIN(chunk_start + chunk_size - 1, seq_len - 1)
        
        ' Extract query chunk
        InitMatrix(q_chunk, chunk_end - chunk_start + 1, head_dim)
        FOR i = chunk_start TO chunk_end
            FOR j = 0 TO head_dim - 1
                q_chunk.data(i - chunk_start, j) = Q.data(i, j)
            NEXT j
        NEXT i
        
        ' Process attention for this chunk
        InitMatrix(output_chunk, q_chunk.rows, head_dim)
        
        IF mask_type = MASK_CAUSAL THEN
            ' For causal attention, we can use a modified K and V
            DIM k_chunk AS Matrix, v_chunk AS Matrix
            
            ' Only use keys and values up to the current position
            InitMatrix(k_chunk, chunk_end + 1, head_dim)
            InitMatrix(v_chunk, chunk_end + 1, head_dim)
            
            FOR i = 0 TO chunk_end
                FOR j = 0 TO head_dim - 1
                    k_chunk.data(i, j) = K.data(i, j)
                    v_chunk.data(i, j) = V.data(i, j)
                NEXT j
            NEXT i
            
            ' Compute attention for this chunk
            StandardAttention(q_chunk, k_chunk, v_chunk, output_chunk, mask_type)
            
            ' Clean up
            FreeMatrix(k_chunk)
            FreeMatrix(v_chunk)
        ELSE
            ' For full attention, use the complete K and V matrices
            StandardAttention(q_chunk, K, V, output_chunk, mask_type)
        END IF
        
        ' Copy result to output matrix
        FOR i = 0 TO output_chunk.rows - 1
            FOR j = 0 TO head_dim - 1
                output.data(chunk_start + i, j) = output_chunk.data(i, j)
            NEXT j
        NEXT i
        
        ' Clean up
        FreeMatrix(q_chunk)
        FreeMatrix(output_chunk)
    NEXT chunk_start
END SUB
```

## Matrix Allocation and Management

The matrix data structures and operations are optimized for memory efficiency:

```basic
' Matrix data structure
TYPE Matrix
    rows AS INTEGER      ' Number of rows
    cols AS INTEGER      ' Number of columns
    data AS INTEGER PTR  ' Pointer to LogQuantized data
    pool_idx AS INTEGER  ' Index in matrix pool, or 0 if not from pool
END TYPE

' Initialize a matrix with the given dimensions
SUB InitMatrix(BYREF matrix AS Matrix, rows AS INTEGER, cols AS INTEGER)
    ' Check if we can get this matrix from the pool
    IF g_matrix_pool.pool_entries > 0 THEN
        matrix.pool_idx = GetMatrixFromPool(rows, cols)
        
        ' If we got a matrix from the pool, we're done
        IF matrix.pool_idx > 0 THEN
            matrix.rows = rows
            matrix.cols = cols
            matrix.data = g_matrix_pool.matrices(matrix.pool_idx).data
            RETURN
        END IF
    END IF
    
    ' Allocate memory for the matrix directly
    matrix.rows = rows
    matrix.cols = cols
    matrix.data = TrackedAllocate(rows * cols * 4, "Matrix", 0, 0)
    matrix.pool_idx = 0
    
    ' Initialize to zero
    ClearMatrix(matrix)
END SUB

' Free a matrix's memory
SUB FreeMatrix(BYREF matrix AS Matrix)
    ' If this matrix is from the pool, return it to the pool
    IF matrix.pool_idx > 0 THEN
        ReturnMatrixToPool(matrix.pool_idx)
        matrix.data = NULL
        matrix.pool_idx = 0
        RETURN
    END IF
    
    ' Otherwise, free the memory directly
    IF matrix.data <> NULL THEN
        TrackedDeallocate(matrix.data)
        matrix.data = NULL
    END IF
    
    matrix.rows = 0
    matrix.cols = 0
END SUB

' Clear a matrix (set all elements to zero)
SUB ClearMatrix(BYREF matrix AS Matrix)
    DIM i AS INTEGER, j AS INTEGER
    
    FOR i = 0 TO matrix.rows - 1
        FOR j = 0 TO matrix.cols - 1
            matrix.data[i * matrix.cols + j] = 0
        NEXT j
    NEXT i
END SUB
```

## Memory-Aware System Integration

### Model Loading and Execution

```basic
' Load and execute a transformer model with memory constraints
SUB TransformerForward(input_ids() AS INTEGER, model_file AS STRING, BYREF output_probs() AS SINGLE)
    ' Check if we can fit the entire model in memory
    DIM file_info AS ModelInfo
    GetModelInfo(model_file, file_info)
    
    DIM est_memory AS LONG = EstimateModelMemory(file_info)
    DIM available_memory AS LONG = g_memory_tracker.memory_limit * 0.8 ' Leave 20% buffer
    
    IF est_memory <= available_memory THEN
        ' We can fit the entire model in memory
        FullModelForward(input_ids(), model_file, output_probs())
    ELSE
        ' We need to stream parameters from disk
        StreamingModelForward(input_ids(), model_file, output_probs())
    END IF
END SUB

' Estimate memory required for a model
FUNCTION EstimateModelMemory(file_info AS ModelInfo) AS LONG
    DIM embed_size AS LONG = file_info.vocab_size * file_info.embed_dim * 4
    DIM layer_size AS LONG = (file_info.embed_dim * file_info.embed_dim * 4) * 8 ' Approximate per layer
    DIM output_size AS LONG = file_info.vocab_size * file_info.embed_dim * 4
    
    FUNCTION = embed_size + (file_info.num_layers * layer_size) + output_size
END FUNCTION

' Load and execute model with all parameters in memory
SUB FullModelForward(input_ids() AS INTEGER, model_file AS STRING, BYREF output_probs() AS SINGLE)
    DIM model AS TransformerModel
    
    ' Load entire model into memory
    LoadFullModel(model_file, model)
    
    ' Process input
    DIM hidden AS Matrix
    InitMatrix(hidden, UBOUND(input_ids) - LBOUND(input_ids) + 1, model.embed_dim)
    
    ' Embedding lookup
    EmbedInputTokens(input_ids, model.token_embed, hidden)
    
    ' Add positional embedding
    AddPositionalEncoding(hidden)
    
    ' Process through transformer layers
    DIM i AS INTEGER
    DIM layer_input AS Matrix, layer_output AS Matrix
    layer_input = hidden
    
    FOR i = 0 TO model.num_layers - 1
        ' Initialize layer output
        InitMatrix(layer_output, layer_input.rows, layer_input.cols)
        
        ' Process layer
        TransformerLayer(layer_input, model.layers(i), layer_output)
        
        ' Swap for next layer
        SwapMatrices(layer_input, layer_output)
    NEXT i
    
    ' Project to vocabulary
    DIM logits AS Matrix
    InitMatrix(logits, layer_input.rows, model.vocab_size)
    MatrixMultiply(layer_input, model.output_projection, logits)
    
    ' Convert to probabilities
    REDIM output_probs(LBOUND(input_ids) TO UBOUND(input_ids), 1 TO model.vocab_size)
    MatrixToArray(logits, output_probs())
    
    ' Clean up
    FreeModel(model)
    FreeMatrix(hidden)
    FreeMatrix(layer_input)
    FreeMatrix(layer_output)
    FreeMatrix(logits)
END SUB

' Load and execute model with parameters streamed from disk
SUB StreamingModelForward(input_ids() AS INTEGER, model_file AS STRING, BYREF output_probs() AS SINGLE)
    DIM model_info AS ModelInfo
    
    ' Open model stream
    IF NOT OpenModelStream(model_file) THEN
        PRINT "ERROR: Failed to open model file for streaming"
        RETURN
    END IF
    
    ' Get model info
    GetModelInfoFromStream(g_model_stream, model_info)
    
    ' Allocate matrices for streaming processing
    DIM hidden AS Matrix
    DIM layer_in AS Matrix, layer_out AS Matrix
    DIM attention_qkv AS Matrix, attention_out AS Matrix
    DIM ffn_in AS Matrix, ffn_out AS Matrix
    DIM norm1_scale AS Matrix, norm1_bias AS Matrix
    DIM norm2_scale AS Matrix, norm2_bias AS Matrix
    DIM embed_matrix AS Matrix
    DIM output_proj AS Matrix
    
    ' Initialize hidden state with embedded tokens
    InitMatrix(hidden, UBOUND(input_ids) - LBOUND(input_ids) + 1, model_info.embed_dim)
    
    ' Load embeddings
    LoadEmbeddingsFromStream(g_model_stream, embed_matrix)
    EmbedInputTokens(input_ids, embed_matrix, hidden)
    
    ' Add positional encoding
    AddPositionalEncoding(hidden)
    
    ' Free embedding matrix as we don't need it anymore
    FreeMatrix(embed_matrix)
    
    ' Initialize layer matrices
    InitMatrix(layer_in, hidden.rows, hidden.cols)
    InitMatrix(layer_out, hidden.rows, hidden.cols)
    
    ' Copy hidden to layer_in for first layer
    CopyMatrix(hidden, layer_in)
    
    ' Process through transformer layers
    DIM i AS INTEGER
    FOR i = 0 TO model_info.num_layers - 1
        ' Load this layer's parameters
        LoadLayerParameters(i, attention_qkv, attention_out, ffn_in, ffn_out, norm1_scale, norm1_bias, norm2_scale, norm2_bias)
        
        ' Process layer
        StreamingTransformerLayer(layer_in, attention_qkv, attention_out, ffn_in, ffn_out, norm1_scale, norm1_bias, norm2_scale, norm2_bias, layer_out)
        
        ' Swap for next layer
        SwapMatrices(layer_in, layer_out)
        
        ' Free layer parameters as we don't need them anymore
        FreeMatrix(attention_qkv)
        FreeMatrix(attention_out)
        FreeMatrix(ffn_in)
        FreeMatrix(ffn_out)
        FreeMatrix(norm1_scale)
        FreeMatrix(norm1_bias)
        FreeMatrix(norm2_scale)
        FreeMatrix(norm2_bias)
    NEXT i
    
    ' Load output projection
    LoadOutputProjectionFromStream(g_model_stream, output_proj)
    
    ' Project to vocabulary
    DIM logits AS Matrix
    InitMatrix(logits, layer_in.rows, model_info.vocab_size)
    MatrixMultiply(layer_in, output_proj, logits)
    
    ' Convert to probabilities
    REDIM output_probs(LBOUND(input_ids) TO UBOUND(input_ids), 1 TO model_info.vocab_size)
    MatrixToArray(logits, output_probs())
    
    ' Clean up
    CloseModelStream()
    FreeMatrix(hidden)
    FreeMatrix(layer_in)
    FreeMatrix(layer_out)
    FreeMatrix(output_proj)
    FreeMatrix(logits)
END SUB
```

### Memory-Efficient Generation

```basic
' Generate text with memory-efficient streaming
SUB GenerateText(prompt_ids() AS INTEGER, model_file AS STRING, max_length AS INTEGER, temperature AS SINGLE, BYREF output_ids() AS INTEGER)
    DIM available_memory AS LONG = g_memory_tracker.memory_limit - g_memory_tracker.total_allocated
    DIM prompt_length AS INTEGER = UBOUND(prompt_ids) - LBOUND(prompt_ids) + 1
    
    ' Allocate output array
    REDIM output_ids(1 TO max_length)
    
    ' Copy prompt to output
    DIM i AS INTEGER
    FOR i = 1 TO prompt_length
        output_ids(i) = prompt_ids(LBOUND(prompt_ids) + i - 1)
    NEXT i
    
    ' Check if we need to use memory-efficient mode
    DIM model_info AS ModelInfo
    GetModelInfo(model_file, model_info)
    
    DIM est_memory AS LONG = EstimateModelMemory(model_info)
    
    ' Open model stream
    IF NOT OpenModelStream(model_file) THEN
        PRINT "ERROR: Failed to open model file for generation"
        RETURN
    END IF
    
    ' Generate tokens one by one
    DIM token_idx AS INTEGER = prompt_length + 1
    DIM input_len AS INTEGER
    
    WHILE token_idx <= max_length
        ' Determine how much context to use
        IF est_memory <= available_memory * 0.8 THEN
            ' We can use full context
            input_len = token_idx - 1
        ELSE
            ' Use sliding window of context
            input_len = MIN(token_idx - 1, model_info.max_context_length)
        END IF
        
        ' Extract input tokens
        DIM current_input() AS INTEGER
        REDIM current_input(1 TO input_len)
        
        FOR i = 1 TO input_len
            current_input(i) = output_ids(token_idx - input_len + i - 1)
        NEXT i
        
        ' Get next token
        DIM next_token_probs() AS SINGLE
        StreamingModelForward(current_input(), model_file, next_token_probs())
        
        ' Sample next token (from last position only)
        output_ids(token_idx) = SampleNextToken(next_token_probs(input_len, 1 TO model_info.vocab_size), temperature)
        
        ' Move to next position
        token_idx = token_idx + 1
    WEND
    
    ' Close model stream
    CloseModelStream()
END SUB
```

## Implementation Sequence

The memory management system will be implemented in the following sequence:

1. **Core Memory Tracking**
   - Implement `MemoryTracker` structure and initialization
   - Develop `TrackedAllocate` and `TrackedDeallocate` functions
   - Create memory usage reporting functions

2. **Matrix Memory Pool**
   - Implement `MatrixPool` structure and initialization
   - Develop matrix allocation and deallocation through the pool
   - Create pool management functions

3. **Parameter Streaming System**
   - Define model file format and structures
   - Implement file I/O for parameter streaming
   - Create layer-by-layer loading functions

4. **Memory-Aware Operations**
   - Develop chunked matrix operations
   - Implement memory-adaptive attention mechanisms
   - Create functions that adapt to available memory

5. **Integration with Model Components**
   - Modify transformer components to use memory management
   - Implement streaming model forward pass
   - Create memory-efficient text generation

## Integration Points

The memory management system integrates with other components as follows:

1. **Matrix Operations (matrix_ops.bas)**
   - Memory-aware matrix multiplication and operations
   - Integration with matrix pooling for reuse

2. **Transformer Components (transformer_components.bas)**
   - Memory-adaptive attention mechanisms
   - Streaming layer execution

3. **Model Operations (model.bas)**
   - Parameter streaming from disk
   - Efficient model initialization and execution

4. **SIMD-like Operations (simd_ops.bas)**
   - Memory-efficient packed operations
   - Integration with memory tracking

5. **Block-Sparse Attention (block_sparse.bas)**
   - Memory-optimized sparse representations
   - Dynamic switching based on available memory

## Success Criteria

The memory management system will be considered successful when:

1. **Memory Constraint Compliance**: The system operates within the 32MB RAM limit on 486-era hardware
2. **Efficient Streaming**: Model parameters are loaded and unloaded with minimal memory usage and acceptable performance
3. **Graceful Degradation**: The system automatically adjusts to limited memory by using more efficient representations and algorithms
4. **Comprehensive Tracking**: Memory usage is accurately tracked and reported during operation
5. **No Leaks**: All allocated memory is properly freed, with no memory leaks
6. **Matrix Reuse**: Matrix memory is efficiently pooled and reused to minimize allocation/deallocation overhead
7. **Adaptive Operations**: Operations automatically adjust their algorithms based on available memory
8. **Integration**: Works seamlessly with other system components

## Conclusion

The memory management system is critical for enabling transformer models to run on 486-era hardware with severe memory constraints. Through careful tracking, pooling, streaming, and adaptive algorithms, we can implement a GPT-2-like model within the 32MB RAM limit. The system provides a flexible foundation that adapts to available resources, ensuring optimal performance across a range of hardware configurations.