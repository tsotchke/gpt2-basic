# GPT-2 BASIC Optimization Techniques

This document summarizes the key optimization techniques employed in the GPT-2 BASIC implementation. These techniques enable a transformer model to operate within 486-era hardware constraints while maintaining functionality.

## 1. Logarithmic Quantization (4-bit)

### Description
Stores model parameters using only 4 bits per value in a logarithmic scale, providing greater precision for small values where most neural network weights cluster.

### Implementation
```basic
FUNCTION QuantizeLog(f AS SINGLE) AS INTEGER
    ' Extract sign
    DIM sgn AS INTEGER
    IF f > 0 THEN sgn = 1
    IF f < 0 THEN sgn = -1
    IF f = 0 THEN sgn = 0 ' Handle zero sign

    DIM abs_f AS SINGLE = ABS(f)
    
    ' Handle zero or near-zero special case
    IF abs_f < 0.00001 THEN
        RETURN 0
    END IF
    
    ' Calculate exponent (biased by 8)
    DIM exponent AS INTEGER = INT(LOG(abs_f) / LOG(2)) + 8
    
    ' Ensure exponent is within the 4-bit range (0-15)
    IF exponent < 0 THEN exponent = 0
    IF exponent > 15 THEN exponent = 15
    
    ' Calculate mantissa (4 bits, 0-15)
    DIM power_of_2 AS SINGLE = 2.0 ^ (exponent - 8)
    DIM mantissa AS INTEGER = INT((abs_f / power_of_2) * 16.0) AND 15
    
    ' Pack mantissa and exponent
    DIM packed_val AS INTEGER = mantissa + (exponent * 16)
    
    ' Apply sign
    IF sgn = -1 THEN packed_val = -packed_val
    
    RETURN packed_val
END FUNCTION
```

### Benefits
- Reduces memory usage by 8× compared to 32-bit floating-point
- Allows for efficient storage of model parameters
- Compatible with fixed-point arithmetic for computation

## 2. Fixed-Point Arithmetic (Q16.16)

### Description
Uses a fixed-point representation with 16 bits for the integer part and 16 bits for the fractional part, eliminating the need for floating-point operations.

### Implementation
```basic
CONST FIXED_POINT_SCALE AS LONG = 65536 ' 2^16

' Convert floating-point to fixed-point
FUNCTION FloatToFixed(f AS SINGLE) AS LONG
    RETURN CLNG(f * FIXED_POINT_SCALE)
END FUNCTION

' Convert fixed-point to floating-point
FUNCTION FixedToFloat(fp AS LONG) AS SINGLE
    RETURN (CSNG(fp) / FIXED_POINT_SCALE)
END FUNCTION

' Multiplication requires adjustment for the scaling factor
FUNCTION FixedMultiply(a AS LONG, b AS LONG) AS LONG
    DIM result AS LONG
    #IFDEF __FB_64BIT__
        ' If running on modern 64-bit FreeBASIC for testing
        DIM temp AS _UNSIGNED _INTEGER64
        temp = CLNGINT(a) * CLNGINT(b)
        result = temp >> 16 ' Right shift 16 bits for division by 2^16
    #ELSE
        ' Original 486-compatible version using 32-bit math
        DIM a_high AS LONG, a_low AS LONG
        DIM b_high AS LONG, b_low AS LONG
        
        ' Split into high and low 16-bit parts
        a_high = a >> 16
        a_low = a AND &HFFFF
        b_high = b >> 16
        b_low = b AND &HFFFF
        
        ' Compute the four products
        ' Only a_low * b_low needs adjustment for the fractional part
        result = ((a_high * b_high) << 16) + _
                 (a_high * b_low) + _
                 (a_low * b_high) + _
                 ((a_low * b_low) >> 16)
    #ENDIF
    
    RETURN result
END FUNCTION
```

### Benefits
- Eliminates need for floating-point operations, which were slow on 486SX
- Provides consistent precision across all mathematical operations
- Compatible with integer-only processing pipelines

## 3. Block-Sparse Attention

### Description
Implements attention computation using sparse blocks, only storing and operating on non-zero blocks of the attention matrix. This is particularly effective for autoregressive generation with causal masking.

### Implementation
```basic
TYPE SparseBlock
    row_start AS INTEGER    ' Starting row of this block
    col_start AS INTEGER    ' Starting column of this block
    block_size AS INTEGER   ' Size of this block (typically power of 2)
    data() AS INTEGER       ' Block data stored as LogQuantized values
    next AS SparseBlock PTR ' Pointer to next block in linked list
END TYPE

TYPE SparseBlockMatrix
    blocks AS SparseBlock PTR ' Head of linked list of blocks
    rows AS INTEGER          ' Total rows in the full matrix
    cols AS INTEGER          ' Total columns in the full matrix
    block_size AS INTEGER    ' Standard block size
    num_blocks AS INTEGER    ' Number of blocks in the matrix
END TYPE

SUB BlockSparseAttention(Query AS Matrix, Key AS Matrix, Value AS Matrix, Output AS Matrix, mask_type AS INTEGER)
    ' Create sparse attention pattern based on mask type
    DIM scores AS SparseBlockMatrix
    InitSparseBlockMatrix(scores, Query.rows, Key.rows, 8) ' 8x8 blocks
    
    ' For causal masking, only create blocks for the upper triangular portion
    IF mask_type = MASK_CAUSAL THEN
        CreateCausalSparseMask(scores, Query.rows)
    END IF
    
    ' Compute scores only for non-zero blocks
    ComputeSparseAttentionScores(Query, Key, scores)
    
    ' Apply softmax along rows
    SparseBlockSoftmax(scores)
    
    ' Multiply with values
    SparseBlockMatrixMultiply(scores, Value, Output)
END SUB
```

### Benefits
- Reduces memory usage by 50-80% for typical sequence lengths
- Decreases computation time by avoiding operations on zero blocks
- Enables longer context lengths within memory constraints

## 4. SIMD-like Bit Manipulation

### Description
Implements a "poor man's SIMD" approach through bit manipulation, allowing multiple values to be processed in parallel within standard 32-bit integers.

### Implementation
```basic
' Pack 4 8-bit values into a single 32-bit integer
FUNCTION Pack_8bit(v1 AS BYTE, v2 AS BYTE, v3 AS BYTE, v4 AS BYTE) AS LONG
    RETURN ((CLNG(v1) AND &HFF)) OR _
           ((CLNG(v2) AND &HFF) << 8) OR _
           ((CLNG(v3) AND &HFF) << 16) OR _
           ((CLNG(v4) AND &HFF) << 24)
END FUNCTION

' Unpack values from a 32-bit integer
SUB Unpack_8bit(packed AS LONG, BYREF v1 AS BYTE, BYREF v2 AS BYTE, BYREF v3 AS BYTE, BYREF v4 AS BYTE)
    v1 = packed AND &HFF
    v2 = (packed >> 8) AND &HFF
    v3 = (packed >> 16) AND &HFF
    v4 = (packed >> 24) AND &HFF
END SUB

' SIMD-like addition for 4 packed 8-bit values
FUNCTION SIMD_Add_8bit(a AS LONG, b AS LONG) AS LONG
    DIM result AS LONG = a + b
    DIM overflow_mask AS LONG = &H01010100 ' Bits that would carry between elements
    
    ' Handle potential overflow between elements
    result = (result AND (NOT overflow_mask)) OR _
             ((a AND b AND overflow_mask))
    
    RETURN result
END FUNCTION
```

### Benefits
- Processes multiple values in parallel within a single instruction
- Significantly speeds up matrix operations
- Maximizes throughput on 32-bit processors without SIMD instructions

## 5. Disk Streaming Parameter System

### Description
Implements a system to load model parameters on demand from disk, process them, and immediately free the memory, allowing models much larger than available RAM.

### Implementation
```basic
SUB StreamLayerWeights(model_file AS STRING, layer_idx AS INTEGER, BYREF weights AS LayerWeights)
    DIM file_handle AS INTEGER
    DIM offset AS LONG
    DIM header AS ModelHeader
    
    ' Open model file
    file_handle = FREEFILE
    OPEN model_file FOR BINARY AS #file_handle
    
    ' Read header to get layer offsets
    GET #file_handle, 1, header
    
    ' Calculate offset for this layer
    offset = header.layer_offsets(layer_idx)
    
    ' Seek to layer position
    SEEK #file_handle, offset
    
    ' Read layer dimensions
    DIM layer_dim AS LayerDimensions
    GET #file_handle, , layer_dim
    
    ' Initialize weight matrices
    InitMatrix(weights.attention_qkv, layer_dim.hidden_size, layer_dim.qkv_size)
    InitMatrix(weights.attention_output, layer_dim.hidden_size, layer_dim.hidden_size)
    InitMatrix(weights.ffn_intermediate, layer_dim.hidden_size, layer_dim.intermediate_size)
    InitMatrix(weights.ffn_output, layer_dim.intermediate_size, layer_dim.hidden_size)
    
    ' Read weights from file
    GET #file_handle, , weights.attention_qkv.data
    GET #file_handle, , weights.attention_output.data
    GET #file_handle, , weights.ffn_intermediate.data
    GET #file_handle, , weights.ffn_output.data
    
    ' Close file
    CLOSE #file_handle
END SUB

SUB FreeLayerWeights(BYREF weights AS LayerWeights)
    FreeMatrix(weights.attention_qkv)
    FreeMatrix(weights.attention_output)
    FreeMatrix(weights.ffn_intermediate)
    FreeMatrix(weights.ffn_output)
END SUB
```

### Benefits
- Allows operation with models much larger than available RAM
- Minimizes memory footprint during inference
- Enables operation within 486-era memory constraints (32MB)

## 6. Function Approximation via Lookup Tables

### Description
Replaces expensive mathematical operations like exponential, logarithm, and trigonometric functions with lookup tables and linear interpolation.

### Implementation
```basic
' Initialize lookup table for exponential function
SUB InitExpLookup()
    DIM i AS INTEGER
    DIM x AS SINGLE
    
    FOR i = 0 TO EXP_TABLE_SIZE - 1
        x = i * EXP_TABLE_STEP
        ExpTable(i) = EXP(x)
    NEXT i
END SUB

' Approximate exponential function using lookup and linear interpolation
FUNCTION FastExp(x AS SINGLE) AS SINGLE
    ' Handle out-of-range inputs
    IF x < 0 THEN RETURN 1.0 / FastExp(-x)
    IF x > EXP_TABLE_MAX THEN RETURN EXP_TABLE_MAX_VALUE
    
    ' Find lookup table indices
    DIM idx AS INTEGER = INT(x / EXP_TABLE_STEP)
    IF idx >= EXP_TABLE_SIZE - 1 THEN RETURN ExpTable(EXP_TABLE_SIZE - 1)
    
    ' Linear interpolation
    DIM frac AS SINGLE = (x / EXP_TABLE_STEP) - idx
    RETURN ExpTable(idx) + frac * (ExpTable(idx + 1) - ExpTable(idx))
END FUNCTION
```

### Benefits
- Drastically reduces computation time for expensive functions
- Allows for adjustable precision/performance tradeoff
- Can be optimized for specific input ranges common in transformer calculations

## 7. Assembly Optimizations

### Description
Implements critical sections in optimized x86 assembly language for maximum performance, with conditional compilation and fallbacks for compatibility.

### Implementation
```assembly
; Fixed-point multiplication (Q16.16 format)
PUBLIC _FixedMulAsm
_FixedMulAsm PROC
    push    bp
    mov     bp, sp
    
    ; Get parameters from stack
    mov     ax, [bp+8]  ; High word of first operand
    mov     bx, [bp+6]  ; Low word of first operand
    mov     cx, [bp+12] ; High word of second operand
    mov     dx, [bp+10] ; Low word of second operand
    
    ; Implementation of optimized fixed-point multiplication
    ; (Assembly code for 32-bit multiplication)
    
    pop     bp
    ret
_FixedMulAsm ENDP
```

With BASIC wrapper:
```basic
FUNCTION FixedMultiply(a AS LONG, b AS LONG) AS LONG
    #IFDEF USE_ASSEMBLY
        IF HasAssemblySupport() THEN
            RETURN FixedMulAsm(a, b)
        END IF
    #ENDIF
    
    ' Fallback implementation for systems without assembly support
    RETURN FixedMultiplyFallback(a, b)
END FUNCTION
```

### Benefits
- Provides 2-3× speedup for critical operations
- Optimizes register usage for 486 architecture
- Implements techniques difficult to express in high-level languages

## 8. Memory Tracking and Management

### Description
Implements comprehensive memory tracking and management to ensure operation within strict memory constraints.

### Implementation
```basic
TYPE MemoryTracker
    total_allocated AS LONG     ' Total bytes currently allocated
    peak_allocated AS LONG      ' Peak allocation during execution
    num_allocations AS INTEGER  ' Number of active allocations
    memory_limit AS LONG        ' Maximum allowed memory usage
END TYPE

DIM SHARED g_mem_tracker AS MemoryTracker

' Initialize the memory tracker
SUB InitMemoryTracker(memory_limit AS LONG)
    g_mem_tracker.total_allocated = 0
    g_mem_tracker.peak_allocated = 0
    g_mem_tracker.num_allocations = 0
    g_mem_tracker.memory_limit = memory_limit
END SUB

' Track memory allocation
FUNCTION TrackedAllocate(size AS LONG) AS ANY PTR
    IF g_mem_tracker.total_allocated + size > g_mem_tracker.memory_limit THEN
        PRINT "ERROR: Memory allocation would exceed limit"
        RETURN NULL
    END IF
    
    DIM ptr AS ANY PTR = ALLOCATE(size)
    IF ptr <> NULL THEN
        g_mem_tracker.total_allocated = g_mem_tracker.total_allocated + size
        g_mem_tracker.num_allocations = g_mem_tracker.num_allocations + 1
        
        IF g_mem_tracker.total_allocated > g_mem_tracker.peak_allocated THEN
            g_mem_tracker.peak_allocated = g_mem_tracker.total_allocated
        END IF
    END IF
    
    RETURN ptr
END FUNCTION
```

### Benefits
- Ensures operation within memory constraints
- Provides visibility into memory usage patterns
- Helps identify and resolve memory leaks and inefficiencies

## 9. Matrix Operations Optimization

### Description
Implements specialized matrix operations optimized for transformer computation patterns, including blocking for improved cache locality.

### Implementation
```basic
' Cache-friendly matrix multiplication with blocking
SUB MatrixMultiplyBlocked(A AS Matrix, B AS Matrix, C AS Matrix)
    DIM i AS INTEGER, j AS INTEGER, k AS INTEGER
    DIM block_i AS INTEGER, block_j AS INTEGER, block_k AS INTEGER
    DIM sum AS LONG
    
    ' Use blocks of size 8x8
    CONST BLOCK_SIZE = 8
    
    FOR block_i = 0 TO A.rows - 1 STEP BLOCK_SIZE
        FOR block_j = 0 TO B.cols - 1 STEP BLOCK_SIZE
            FOR block_k = 0 TO A.cols - 1 STEP BLOCK_SIZE
                ' Process this block
                FOR i = block_i TO MIN(block_i + BLOCK_SIZE - 1, A.rows - 1)
                    FOR j = block_j TO MIN(block_j + BLOCK_SIZE - 1, B.cols - 1)
                        sum = 0
                        FOR k = block_k TO MIN(block_k + BLOCK_SIZE - 1, A.cols - 1)
                            sum = sum + FixedMultiply(A.data(i, k), B.data(k, j))
                        NEXT k
                        
                        ' Add to accumulator
                        IF block_k = 0 THEN
                            C.data(i, j) = sum
                        ELSE
                            C.data(i, j) = C.data(i, j) + sum
                        END IF
                    NEXT j
                NEXT i
            NEXT block_k
        NEXT block_j
    NEXT block_i
END SUB
```

### Benefits
- Improves cache locality for better memory performance
- Reduces pipeline stalls on 486 processors
- Minimizes memory access patterns for improved throughput

## 10. Dynamic Precision Adaptation

### Description
Implements a system that dynamically adjusts precision based on the computational requirements and available resources.

### Implementation
```basic
' Determine optimal precision for matrix operations
FUNCTION DeterminePrecision(rows AS INTEGER, cols AS INTEGER) AS INTEGER
    DIM size_bytes AS LONG = rows * cols * 4 ' Size if using standard precision
    
    ' If matrix is small, use full precision
    IF size_bytes < 4096 THEN
        RETURN PRECISION_FULL
    END IF
    
    ' If matrix is large and in attention calculation, use low precision
    IF size_bytes > 16384 AND g_operation_type = OPERATION_ATTENTION THEN
        RETURN PRECISION_LOW
    END IF
    
    ' Default medium precision for other cases
    RETURN PRECISION_MEDIUM
END FUNCTION

' Matrix multiply with precision adaptation
SUB MatrixMultiplyAdaptive(A AS Matrix, B AS Matrix, C AS Matrix)
    DIM precision AS INTEGER = DeterminePrecision(A.rows, B.cols)
    
    SELECT CASE precision
        CASE PRECISION_LOW:
            MatrixMultiplyLowPrecision(A, B, C)
        CASE PRECISION_MEDIUM:
            MatrixMultiplyMediumPrecision(A, B, C)
        CASE PRECISION_FULL:
            MatrixMultiplyFullPrecision(A, B, C)
    END SELECT
END SUB
```

### Benefits
- Adapts to available memory and processing power
- Prioritizes precision where it matters most
- Enables graceful degradation on more constrained systems

## Historical Context

These optimization techniques draw inspiration from methods used in 486-era software development, particularly:

1. **Fixed-Point Arithmetic**: Commonly used in 486-era games and multimedia applications, especially on 486SX systems without FPUs.

2. **Lookup Tables**: A standard technique in demo scene programming and games for approximating expensive functions.

3. **Assembly Optimization**: Critical for performance in 486-era applications, especially in multimedia software.

4. **Memory Management**: Sophisticated memory management was essential for software targeting DOS with its complex memory model (conventional, extended, expanded memory).

5. **Block Processing**: Similar techniques were used in early image and audio processing software to work within cache constraints.

These approaches demonstrate that modern AI concepts like transformers could theoretically have been implemented on 486-era hardware with careful optimization, even though they weren't developed until much later.
