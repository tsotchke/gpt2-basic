' *******************************************************
' * Matrix Operations for GPT-2 BASIC                   *
' *******************************************************
' * This module implements optimized matrix operations  *
' * for the GPT-2 model, designed for 486-era hardware  *
' * constraints.                                        *
' *                                                     *
' * It provides basic and advanced matrix operations,   *
' * with performance optimizations to maximize speed    *
' * while minimizing memory usage.                      *
' *******************************************************

#INCLUDE "src/data_structures.bas"
#INCLUDE "src/simd_ops.bas"

' *******************************************************
' * Configuration and Performance Tracking              *
' *******************************************************

' Optimization level (0=none, 1=basic, 2=advanced, 3=max)
DIM SHARED g_optimization_level AS INTEGER

' Matrix operation count for performance tracking
DIM SHARED g_matrix_mul_count AS LONG
DIM SHARED g_matrix_add_count AS LONG
DIM SHARED g_matrix_scale_count AS LONG
DIM SHARED g_matrix_dot_count AS LONG

' Initialize the matrix operations system
SUB InitMatrixOps()
    ' Reset operation counters
    g_matrix_mul_count = 0
    g_matrix_add_count = 0
    g_matrix_scale_count = 0
    g_matrix_dot_count = 0
    
    ' Set default optimization level
    g_optimization_level = 2 ' Default to advanced optimizations
    
    ' Adjust optimization level based on CPU capabilities
    IF NOT g_cpu_detected THEN
        DetectCPU()
    END IF
    
    SELECT CASE g_cpu_type
        CASE CPU_486SX: g_optimization_level = 1 ' Basic optimizations for 486SX
        CASE CPU_486DX, CPU_486DX2: g_optimization_level = 2 ' Advanced for 486DX/DX2
        CASE CPU_486DX4, CPU_PENTIUM: g_optimization_level = 3 ' Maximum for DX4/Pentium
    END SELECT
    
    PRINT "Matrix operations initialized with optimization level "; g_optimization_level
END SUB

' Print matrix operations statistics
SUB PrintMatrixStats()
    PRINT "Matrix Operations Statistics:"
    PRINT "  Matrix multiplications: "; g_matrix_mul_count
    PRINT "  Matrix additions: "; g_matrix_add_count
    PRINT "  Matrix scaling operations: "; g_matrix_scale_count
    PRINT "  Dot product operations: "; g_matrix_dot_count
END SUB

' *******************************************************
' * Basic Matrix Operations                             *
' *******************************************************

' Matrix multiplication: C = A * B
SUB MatrixMultiply(A AS Matrix, B AS Matrix, BYREF C AS Matrix)
    DIM i AS INTEGER, j AS INTEGER, k AS INTEGER
    DIM sum AS SINGLE
    
    ' Ensure dimensions are compatible
    IF A.cols <> B.rows THEN
        PRINT "ERROR: Matrix dimensions incompatible for multiplication"
        RETURN
    END IF
    
    ' Initialize result matrix if needed
    IF C.rows <> A.rows OR C.cols <> B.cols THEN
        FreeMatrix(C)
        InitMatrix(C, A.rows, B.cols)
    END IF
    
    ' Choose implementation based on optimization level
    SELECT CASE g_optimization_level
        CASE 0: ' No optimization
            ' Basic implementation
            FOR i = 0 TO A.rows - 1
                FOR j = 0 TO B.cols - 1
                    sum = 0.0
                    FOR k = 0 TO A.cols - 1
                        sum = sum + A.data(i, k) * B.data(k, j)
                    NEXT k
                    C.data(i, j) = sum
                NEXT j
            NEXT i
            
        CASE 1: ' Basic optimization
            ' Cache-friendly loop ordering (ikj instead of ijk)
            FOR i = 0 TO A.rows - 1
                FOR k = 0 TO A.cols - 1
                    FOR j = 0 TO B.cols - 1
                        C.data(i, j) = C.data(i, j) + A.data(i, k) * B.data(k, j)
                    NEXT j
                NEXT k
            NEXT i
            
        CASE 2, 3: ' Advanced optimization
            ' Cache blocking for better locality
            DIM block_size AS INTEGER
            block_size = 16 ' Adjust based on cache size
            
            ' Initialize result matrix to zeros
            ZeroMatrix(C)
            
            ' Blocked matrix multiplication
            DIM i_block AS INTEGER, j_block AS INTEGER, k_block AS INTEGER
            
            FOR i_block = 0 TO A.rows - 1 STEP block_size
                FOR j_block = 0 TO B.cols - 1 STEP block_size
                    FOR k_block = 0 TO A.cols - 1 STEP block_size
                        ' Multiply blocks
                        DIM i_end AS INTEGER, j_end AS INTEGER, k_end AS INTEGER
                        i_end = MIN(i_block + block_size - 1, A.rows - 1)
                        j_end = MIN(j_block + block_size - 1, B.cols - 1)
                        k_end = MIN(k_block + block_size - 1, A.cols - 1)
                        
                        FOR i = i_block TO i_end
                            FOR k = k_block TO k_end
                                DIM a_val AS SINGLE
                                a_val = A.data(i, k)
                                
                                FOR j = j_block TO j_end
                                    C.data(i, j) = C.data(i, j) + a_val * B.data(k, j)
                                NEXT j
                            NEXT k
                        NEXT i
                    NEXT k_block
                NEXT j_block
            NEXT i_block
    END SELECT
    
    ' Track operation count
    g_matrix_mul_count = g_matrix_mul_count + 1
END SUB

' Matrix multiplication with transposed second matrix: C = A * B^T
SUB MatrixMultiplyTransposeB(A AS Matrix, B AS Matrix, BYREF C AS Matrix)
    DIM i AS INTEGER, j AS INTEGER, k AS INTEGER
    DIM sum AS SINGLE
    
    ' Ensure dimensions are compatible
    IF A.cols <> B.cols THEN
        PRINT "ERROR: Matrix dimensions incompatible for multiplication with transposed B"
        RETURN
    END IF
    
    ' Initialize result matrix if needed
    IF C.rows <> A.rows OR C.cols <> B.rows THEN
        FreeMatrix(C)
        InitMatrix(C, A.rows, B.rows)
    END IF
    
    ' Choose implementation based on optimization level
    SELECT CASE g_optimization_level
        CASE 0: ' No optimization
            ' Basic implementation
            FOR i = 0 TO A.rows - 1
                FOR j = 0 TO B.rows - 1
                    sum = 0.0
                    FOR k = 0 TO A.cols - 1
                        sum = sum + A.data(i, k) * B.data(j, k)
                    NEXT k
                    C.data(i, j) = sum
                NEXT j
            NEXT i
            
        CASE 1, 2, 3: ' Optimized implementation
            ' Cache-friendly loop ordering and pre-initialization
            ZeroMatrix(C)
            
            FOR i = 0 TO A.rows - 1
                FOR k = 0 TO A.cols - 1
                    DIM a_val AS SINGLE
                    a_val = A.data(i, k)
                    
                    FOR j = 0 TO B.rows - 1
                        C.data(i, j) = C.data(i, j) + a_val * B.data(j, k)
                    NEXT j
                NEXT k
            NEXT i
    END SELECT
    
    ' Track operation count
    g_matrix_mul_count = g_matrix_mul_count + 1
END SUB

' Matrix multiplication with transposed first matrix: C = A^T * B
SUB MatrixMultiplyTransposeA(A AS Matrix, B AS Matrix, BYREF C AS Matrix)
    DIM i AS INTEGER, j AS INTEGER, k AS INTEGER
    DIM sum AS SINGLE
    
    ' Ensure dimensions are compatible
    IF A.rows <> B.rows THEN
        PRINT "ERROR: Matrix dimensions incompatible for multiplication with transposed A"
        RETURN
    END IF
    
    ' Initialize result matrix if needed
    IF C.rows <> A.cols OR C.cols <> B.cols THEN
        FreeMatrix(C)
        InitMatrix(C, A.cols, B.cols)
    END IF
    
    ' Choose implementation based on optimization level
    SELECT CASE g_optimization_level
        CASE 0: ' No optimization
            ' Basic implementation
            FOR i = 0 TO A.cols - 1
                FOR j = 0 TO B.cols - 1
                    sum = 0.0
                    FOR k = 0 TO A.rows - 1
                        sum = sum + A.data(k, i) * B.data(k, j)
                    NEXT k
                    C.data(i, j) = sum
                NEXT j
            NEXT i
            
        CASE 1, 2, 3: ' Optimized implementation
            ' Cache-friendly method
            ZeroMatrix(C)
            
            FOR k = 0 TO A.rows - 1
                FOR i = 0 TO A.cols - 1
                    DIM a_val AS SINGLE
                    a_val = A.data(k, i)
                    
                    FOR j = 0 TO B.cols - 1
                        C.data(i, j) = C.data(i, j) + a_val * B.data(k, j)
                    NEXT j
                NEXT i
            NEXT k
    END SELECT
    
    ' Track operation count
    g_matrix_mul_count = g_matrix_mul_count + 1
END SUB

' Matrix addition: C = A + B
SUB MatrixAdd(A AS Matrix, B AS Matrix, BYREF C AS Matrix)
    DIM i AS INTEGER, j AS INTEGER
    
    ' Ensure dimensions are compatible
    IF A.rows <> B.rows OR A.cols <> B.cols THEN
        PRINT "ERROR: Matrix dimensions incompatible for addition"
        RETURN
    END IF
    
    ' Initialize result matrix if needed
    IF C.rows <> A.rows OR C.cols <> A.cols THEN
        FreeMatrix(C)
        InitMatrix(C, A.rows, A.cols)
    END IF
    
    ' Add matrices
    FOR i = 0 TO A.rows - 1
        FOR j = 0 TO A.cols - 1
            C.data(i, j) = A.data(i, j) + B.data(i, j)
        NEXT j
    NEXT i
    
    ' Track operation count
    g_matrix_add_count = g_matrix_add_count + 1
END SUB

' Matrix subtraction: C = A - B
SUB MatrixSubtract(A AS Matrix, B AS Matrix, BYREF C AS Matrix)
    DIM i AS INTEGER, j AS INTEGER
    
    ' Ensure dimensions are compatible
    IF A.rows <> B.rows OR A.cols <> B.cols THEN
        PRINT "ERROR: Matrix dimensions incompatible for subtraction"
        RETURN
    END IF
    
    ' Initialize result matrix if needed
    IF C.rows <> A.rows OR C.cols <> A.cols THEN
        FreeMatrix(C)
        InitMatrix(C, A.rows, A.cols)
    END IF
    
    ' Subtract matrices
    FOR i = 0 TO A.rows - 1
        FOR j = 0 TO A.cols - 1
            C.data(i, j) = A.data(i, j) - B.data(i, j)
        NEXT j
    NEXT i
    
    ' Track operation count
    g_matrix_add_count = g_matrix_add_count + 1
END SUB

' Matrix element-wise multiplication (Hadamard product): C = A .* B
SUB MatrixElementwiseMultiply(A AS Matrix, B AS Matrix, BYREF C AS Matrix)
    DIM i AS INTEGER, j AS INTEGER
    
    ' Ensure dimensions are compatible
    IF A.rows <> B.rows OR A.cols <> B.cols THEN
        PRINT "ERROR: Matrix dimensions incompatible for element-wise multiplication"
        RETURN
    END IF
    
    ' Initialize result matrix if needed
    IF C.rows <> A.rows OR C.cols <> A.cols THEN
        FreeMatrix(C)
        InitMatrix(C, A.rows, A.cols)
    END IF
    
    ' Multiply matrices element-wise
    FOR i = 0 TO A.rows - 1
        FOR j = 0 TO A.cols - 1
            C.data(i, j) = A.data(i, j) * B.data(i, j)
        NEXT j
    NEXT i
    
    ' Track operation count
    g_matrix_mul_count = g_matrix_mul_count + 1
END SUB

' Matrix scaling: B = alpha * A
SUB MatrixScale(A AS Matrix, alpha AS SINGLE, BYREF B AS Matrix)
    DIM i AS INTEGER, j AS INTEGER
    
    ' Initialize result matrix if needed
    IF B.rows <> A.rows OR B.cols <> A.cols THEN
        FreeMatrix(B)
        InitMatrix(B, A.rows, A.cols)
    END IF
    
    ' Scale matrix
    FOR i = 0 TO A.rows - 1
        FOR j = 0 TO A.cols - 1
            B.data(i, j) = alpha * A.data(i, j)
        NEXT j
    NEXT i
    
    ' Track operation count
    g_matrix_scale_count = g_matrix_scale_count + 1
END SUB

' *******************************************************
' * Matrix Operations with SIMD-like Optimizations      *
' *******************************************************

' Matrix multiplication using SIMD-like optimizations
SUB MatrixMultiplySIMD(A AS Matrix, B AS Matrix, BYREF C AS Matrix)
    DIM i AS INTEGER, j AS INTEGER, k AS INTEGER
    
    ' Ensure dimensions are compatible
    IF A.cols <> B.rows THEN
        PRINT "ERROR: Matrix dimensions incompatible for multiplication"
        RETURN
    END IF
    
    ' Initialize result matrix if needed
    IF C.rows <> A.rows OR C.cols <> B.cols THEN
        FreeMatrix(C)
        InitMatrix(C, A.rows, B.cols)
    END IF
    
    ' Choose precision level based on matrix size
    DIM precision AS PrecisionLevel
    precision = DetermineOptimalPrecision(A.rows * A.cols, OPERATION_GENERAL)
    
    ' Choose implementation based on precision
    SELECT CASE precision
        CASE PRECISION_8BIT:
            ' Convert matrices to 8-bit precision
            DIM A_8bit() AS BYTE, B_8bit() AS BYTE, C_8bit() AS BYTE
            REDIM A_8bit(0 TO A.rows * A.cols - 1)
            REDIM B_8bit(0 TO B.rows * B.cols - 1)
            REDIM C_8bit(0 TO A.rows * B.cols - 1)
            
            ' Fill 8-bit matrices (simplified quantization)
            DIM idx AS INTEGER
            idx = 0
            FOR i = 0 TO A.rows - 1
                FOR j = 0 TO A.cols - 1
                    ' Convert float to byte (0-255)
                    A_8bit(idx) = MAX(0, MIN(255, INT((A.data(i, j) + 1.0) * 127.5)))
                    idx = idx + 1
                NEXT j
            NEXT i
            
            idx = 0
            FOR i = 0 TO B.rows - 1
                FOR j = 0 TO B.cols - 1
                    ' Convert float to byte (0-255)
                    B_8bit(idx) = MAX(0, MIN(255, INT((B.data(i, j) + 1.0) * 127.5)))
                    idx = idx + 1
                NEXT j
            NEXT i
            
            ' Use 8-bit SIMD matrix multiplication
            MatrixMultiplySIMD_8bit(A_8bit(), B_8bit(), C_8bit(), A.rows, A.cols, B.cols)
            
            ' Convert result back to float
            idx = 0
            FOR i = 0 TO A.rows - 1
                FOR j = 0 TO B.cols - 1
                    ' Convert byte to float (-1.0 to 1.0)
                    C.data(i, j) = (C_8bit(idx) / 127.5) - 1.0
                    idx = idx + 1
                NEXT j
            NEXT i
            
        CASE PRECISION_4BIT:
            ' Implement native 4-bit version for 486-class systems
            ' This version packs 2 4-bit values per byte for memory efficiency
            
            ' Convert matrices to 4-bit precision
            DIM A_4bit() AS BYTE, B_4bit() AS BYTE, C_4bit() AS BYTE
            ' Each byte holds 2 4-bit values, so we need half the storage
            REDIM A_4bit(0 TO (A.rows * A.cols + 1) \ 2 - 1)
            REDIM B_4bit(0 TO (B.rows * B.cols + 1) \ 2 - 1)
            REDIM C_4bit(0 TO (A.rows * B.cols + 1) \ 2 - 1)
            
            ' Fill 4-bit matrices (quantization)
            DIM idx AS INTEGER, packed_idx AS INTEGER
            idx = 0
            FOR i = 0 TO A.rows - 1
                FOR j = 0 TO A.cols - 1
                    ' Convert float to 4-bit value (0-15)
                    DIM val4bit AS BYTE
                    val4bit = MAX(0, MIN(15, INT((A.data(i, j) + 1.0) * 7.5)))
                    
                    ' Pack 2 values per byte
                    packed_idx = idx \ 2
                    IF idx MOD 2 = 0 THEN
                        ' First 4-bit value goes in the lower bits
                        A_4bit(packed_idx) = val4bit
                    ELSE
                        ' Second 4-bit value goes in the upper bits
                        A_4bit(packed_idx) = A_4bit(packed_idx) OR (val4bit << 4)
                    END IF
                    idx = idx + 1
                NEXT j
            NEXT i
            
            idx = 0
            FOR i = 0 TO B.rows - 1
                FOR j = 0 TO B.cols - 1
                    ' Convert float to 4-bit value (0-15)
                    DIM val4bit AS BYTE
                    val4bit = MAX(0, MIN(15, INT((B.data(i, j) + 1.0) * 7.5)))
                    
                    ' Pack 2 values per byte
                    packed_idx = idx \ 2
                    IF idx MOD 2 = 0 THEN
                        ' First 4-bit value goes in the lower bits
                        B_4bit(packed_idx) = val4bit
                    ELSE
                        ' Second 4-bit value goes in the upper bits
                        B_4bit(packed_idx) = B_4bit(packed_idx) OR (val4bit << 4)
                    END IF
                    idx = idx + 1
                NEXT j
            NEXT i
            
            ' Use specialized 4-bit SIMD-like matrix multiplication
            MatrixMultiplySIMD_4bit(A_4bit(), B_4bit(), C_4bit(), A.rows, A.cols, B.cols)
            
            ' Convert result back to float
            idx = 0
            FOR i = 0 TO A.rows - 1
                FOR j = 0 TO B.cols - 1
                    ' Extract 4-bit value and convert to float (-1.0 to 1.0)
                    packed_idx = idx \ 2
                    DIM val4bit AS BYTE
                    
                    IF idx MOD 2 = 0 THEN
                        ' Extract from lower 4 bits
                        val4bit = C_4bit(packed_idx) AND &H0F
                    ELSE
                        ' Extract from upper 4 bits
                        val4bit = (C_4bit(packed_idx) >> 4) AND &H0F
                    END IF
                    
                    C.data(i, j) = (val4bit / 7.5) - 1.0
                    idx = idx + 1
                NEXT j
            NEXT i
            
        CASE PRECISION_16BIT:
            ' Convert matrices to 16-bit precision (use integers)
            DIM A_16bit() AS INTEGER, B_16bit() AS INTEGER, C_16bit() AS INTEGER
            REDIM A_16bit(0 TO A.rows * A.cols - 1)
            REDIM B_16bit(0 TO B.rows * B.cols - 1)
            REDIM C_16bit(0 TO A.rows * B.cols - 1)
            
            ' Fill 16-bit matrices
            DIM idx AS INTEGER
            idx = 0
            FOR i = 0 TO A.rows - 1
                FOR j = 0 TO A.cols - 1
                    ' Convert float to 16-bit integer (-32768 to 32767)
                    A_16bit(idx) = MAX(-32768, MIN(32767, INT(A.data(i, j) * 32767.0)))
                    idx = idx + 1
                NEXT j
            NEXT i
            
            idx = 0
            FOR i = 0 TO B.rows - 1
                FOR j = 0 TO B.cols - 1
                    ' Convert float to 16-bit integer
                    B_16bit(idx) = MAX(-32768, MIN(32767, INT(B.data(i, j) * 32767.0)))
                    idx = idx + 1
                NEXT j
            NEXT i
            
            ' Implement native 16-bit matrix multiplication with SIMD-like optimizations
            ' This provides higher precision while still using SIMD-like techniques
            
            ' Use optimized dot product for better performance
            idx = 0
            FOR i = 0 TO A.rows - 1
                FOR j = 0 TO B.cols - 1
                    ' Use specialized SIMD-like dot product function for 16-bit values
                    DIM result AS LONG
                    result = DotProduct_16bit(A_16bit, B_16bit, i * A.cols, j, A.cols)
                    
                    ' Scale result appropriately (result is already normalized by the dot product function)
                    C_16bit(idx) = MAX(-32768, MIN(32767, result))
                    idx = idx + 1
                NEXT j
            NEXT i
            
            ' Convert result back to float
            idx = 0
            FOR i = 0 TO A.rows - 1
                FOR j = 0 TO B.cols - 1
                    ' Convert 16-bit integer to float
                    C.data(i, j) = C_16bit(idx) / 32767.0
                    idx = idx + 1
                NEXT j
            NEXT i
            
        CASE PRECISION_32BIT:
            ' Just use standard matrix multiplication for full precision
            MatrixMultiply(A, B, C)
    END SELECT
    
    ' Track operation count
    g_matrix_mul_count = g_matrix_mul_count + 1
END SUB

' Matrix addition using SIMD-like optimizations
SUB MatrixAddSIMD(A AS Matrix, B AS Matrix, BYREF C AS Matrix)
    DIM i AS INTEGER, j AS INTEGER
    
    ' Ensure dimensions are compatible
    IF A.rows <> B.rows OR A.cols <> B.cols THEN
        PRINT "ERROR: Matrix dimensions incompatible for addition"
        RETURN
    END IF
    
    ' Initialize result matrix if needed
    IF C.rows <> A.rows OR C.cols <> A.cols THEN
        FreeMatrix(C)
        InitMatrix(C, A.rows, A.cols)
    END IF
    
    ' Determine if we should use SIMD operations
    IF g_optimization_level >= 2 THEN
        ' Choose precision level
        DIM precision AS PrecisionLevel
        precision = DetermineOptimalPrecision(A.rows * A.cols, OPERATION_GENERAL)
        
        SELECT CASE precision
            CASE PRECISION_8BIT:
                ' Convert to 8-bit and use SIMD addition
                DIM A_8bit() AS BYTE, B_8bit() AS BYTE, C_8bit() AS BYTE
                REDIM A_8bit(0 TO A.rows * A.cols - 1)
                REDIM B_8bit(0 TO B.rows * B.cols - 1)
                REDIM C_8bit(0 TO A.rows * B.cols - 1)
                
                ' Convert to 8-bit format
                DIM idx AS INTEGER
                idx = 0
                FOR i = 0 TO A.rows - 1
                    FOR j = 0 TO A.cols - 1
                        A_8bit(idx) = MAX(0, MIN(255, INT((A.data(i, j) + 1.0) * 127.5)))
                        B_8bit(idx) = MAX(0, MIN(255, INT((B.data(i, j) + 1.0) * 127.5)))
                        idx = idx + 1
                    NEXT j
                NEXT i
                
                ' Process 4 elements at a time with SIMD
                FOR i = 0 TO (A.rows * A.cols - 1) STEP 4
                    IF i + 3 < A.rows * A.cols THEN
                        ' Pack 4 elements
                        DIM a_packed AS LONG, b_packed AS LONG, sum_packed AS LONG
                        a_packed = Pack_8bit(A_8bit(i), A_8bit(i+1), A_8bit(i+2), A_8bit(i+3))
                        b_packed = Pack_8bit(B_8bit(i), B_8bit(i+1), B_8bit(i+2), B_8bit(i+3))
                        
                        ' Add using SIMD
                        sum_packed = SIMD_Add_8bit(a_packed, b_packed)
                        
                        ' Unpack results
                        DIM temp_vals(1 TO 4) AS BYTE
                        Unpack_8bit(sum_packed, temp_vals(1), temp_vals(2), temp_vals(3), temp_vals(4))
                        
                        ' Store results
                        C_8bit(i) = temp_vals(1)
                        C_8bit(i+1) = temp_vals(2)
                        C_8bit(i+2) = temp_vals(3)
                        C_8bit(i+3) = temp_vals(4)
                    ELSE
                        ' Handle remaining elements
                        FOR j = i TO A.rows * A.cols - 1
                            C_8bit(j) = (A_8bit(j) + B_8bit(j)) AND &HFF
                        NEXT j
                    END IF
                NEXT i
                
                ' Convert back to float
                idx = 0
                FOR i = 0 TO A.rows - 1
                    FOR j = 0 TO A.cols - 1
                        C.data(i, j) = (C_8bit(idx) / 127.5) - 1.0
                        idx = idx + 1
                    NEXT j
                NEXT i
                
            CASE ELSE:
                ' Standard addition for other precisions
                FOR i = 0 TO A.rows - 1
                    FOR j = 0 TO A.cols - 1
                        C.data(i, j) = A.data(i, j) + B.data(i, j)
                    NEXT j
                NEXT i
        END SELECT
    ELSE
        ' Standard matrix addition
        FOR i = 0 TO A.rows - 1
            FOR j = 0 TO A.cols - 1
                C.data(i, j) = A.data(i, j) + B.data(i, j)
            NEXT j
        NEXT i
    END IF
    
    ' Track operation count
    g_matrix_add_count = g_matrix_add_count + 1
END SUB

' *******************************************************
' * Advanced Matrix Operations                          *
' *******************************************************

' Matrix transposition: B = A^T
SUB MatrixTranspose(A AS Matrix, BYREF B AS Matrix)
    DIM i AS INTEGER, j AS INTEGER
    
    ' Initialize result matrix if needed
    IF B.rows <> A.cols OR B.cols <> A.rows THEN
        FreeMatrix(B)
        InitMatrix(B, A.cols, A.rows)
    END IF
    
    ' Transpose matrix
    FOR i = 0 TO A.rows - 1
        FOR j = 0 TO A.cols - 1
            B.data(j, i) = A.data(i, j)
        NEXT j
    NEXT i
END SUB

' Matrix row-wise softmax: B_ij = exp(A_ij) / sum_k(exp(A_ik))
SUB MatrixSoftmax(A AS Matrix, BYREF B AS Matrix)
    DIM i AS INTEGER, j AS INTEGER
    DIM row_max AS SINGLE, row_sum AS SINGLE
    
    ' Initialize result matrix if needed
    IF B.rows <> A.rows OR B.cols <> A.cols THEN
        FreeMatrix(B)
        InitMatrix(B, A.rows, A.cols)
    END IF
    
    ' Apply softmax to each row
    FOR i = 0 TO A.rows - 1
        ' Find maximum value in row (for numerical stability)
        row_max = A.data(i, 0)
        FOR j = 1 TO A.cols - 1
            IF A.data(i, j) > row_max THEN
                row_max = A.data(i, j)
            END IF
        NEXT j
        
        ' Compute exp(x - max) and row sum
        row_sum = 0.0
        FOR j = 0 TO A.cols - 1
            B.data(i, j) = EXP(A.data(i, j) - row_max)
            row_sum = row_sum + B.data(i, j)
        NEXT j
        
        ' Normalize by row sum
        FOR j = 0 TO A.cols - 1
            B.data(i, j) = B.data(i, j) / row_sum
        NEXT j
    NEXT i
END SUB

' Compute matrix inverse for small matrices (up to 4x4)
FUNCTION MatrixInverse(A AS Matrix, BYREF B AS Matrix) AS INTEGER
    DIM i AS INTEGER, j AS INTEGER, k AS INTEGER
    DIM determinant AS SINGLE
    
    ' Check dimensions
    IF A.rows <> A.cols THEN
        PRINT "ERROR: Matrix must be square for inversion"
        RETURN 0
    END IF
    
    ' Only support small matrices
    IF A.rows > 4 THEN
        PRINT "ERROR: Matrix inversion only supported for matrices up to 4x4"
        RETURN 0
    END IF
    
    ' Initialize result matrix
    FreeMatrix(B)
    InitMatrix(B, A.rows, A.cols)
    
    ' Special case for 1x1 matrix
    IF A.rows = 1 THEN
        IF ABS(A.data(0, 0)) < 0.000001 THEN
            PRINT "ERROR: Matrix is singular, cannot invert"
            RETURN 0
        END IF
        
        B.data(0, 0) = 1.0 / A.data(0, 0)
        RETURN 1
    END IF
    
    ' Special case for 2x2 matrix
    IF A.rows = 2 THEN
        determinant = A.data(0, 0) * A.data(1, 1) - A.data(0, 1) * A.data(1, 0)
        
        IF ABS(determinant) < 0.000001 THEN
            PRINT "ERROR: Matrix is singular, cannot invert"
            RETURN 0
        END IF
        
        B.data(0, 0) = A.data(1, 1) / determinant
        B.data(0, 1) = -A.data(0, 1) / determinant
        B.data(1, 0) = -A.data(1, 0) / determinant
        B.data(1, 1) = A.data(0, 0) / determinant
        
        RETURN 1
    END IF
    
    ' For 3x3 and 4x4 matrices, use Gauss-Jordan elimination
    
    ' Create augmented matrix [A|I]
    DIM augmented AS Matrix
    InitMatrix(augmented, A.rows, A.cols * 2)
    
    ' Fill augmented matrix
    FOR i = 0 TO A.rows - 1
        FOR j = 0 TO A.cols - 1
            augmented.data(i, j) = A.data(i, j)           ' Left side: A
            augmented.data(i, j + A.cols) = 0.0            ' Right side: I
        NEXT j
        augmented.data(i, i + A.cols) = 1.0                ' Diagonal of I
    NEXT i
    
    ' Perform Gauss-Jordan elimination
    FOR i = 0 TO A.rows - 1
        ' Find pivot
        DIM pivot_row AS INTEGER
        DIM max_val AS SINGLE
        
        max_val = 0.0
        pivot_row = i
        
        FOR k = i TO A.rows - 1
            IF ABS(augmented.data(k, i)) > max_val THEN
                max_val = ABS(augmented.data(k, i))
                pivot_row = k
            END IF
        NEXT k
        
        ' Check if matrix is singular
        IF max_val < 0.000001 THEN
            PRINT "ERROR: Matrix is singular, cannot invert"
            FreeMatrix(augmented)
            RETURN 0
        END IF
        
        ' Swap rows if needed
        IF pivot_row <> i THEN
            FOR j = 0 TO 2 * A.cols - 1
                DIM temp AS SINGLE
                temp = augmented.data(i, j)
                augmented.data(i, j) = augmented.data(pivot_row, j)
                augmented.data(pivot_row, j) = temp
            NEXT j
        END IF
        
        ' Scale pivot row
        DIM pivot_val AS SINGLE
        pivot_val = augmented.data(i, i)
        
        FOR j = 0 TO 2 * A.cols - 1
            augmented.data(i, j) = augmented.data(i, j) / pivot_val
        NEXT j
        
        ' Eliminate other rows
        FOR k = 0 TO A.rows - 1
            IF k <> i THEN
                DIM factor AS SINGLE
                factor = augmented.data(k, i)
                
                FOR j = 0 TO 2 * A.cols - 1
                    augmented.data(k, j) = augmented.data(k, j) - factor * augmented.data(i, j)
                NEXT j
            END IF
        NEXT k
    NEXT i
    
    ' Extract inverse from right half of augmented matrix
    FOR i = 0 TO A.rows - 1
        FOR j = 0 TO A.cols - 1
            B.data(i, j) = augmented.data(i, j + A.cols)
        NEXT j
    NEXT i
    
    ' Free augmented matrix
    FreeMatrix(augmented)
    
    RETURN 1
END FUNCTION

' Apply tanh activation function element-wise
SUB MatrixTanh(A AS Matrix, BYREF B AS Matrix)
    DIM i AS INTEGER, j AS INTEGER
    
    ' Initialize result matrix if needed
    IF B.rows <> A.rows OR B.cols <> A.cols THEN
        FreeMatrix(B)
        InitMatrix(B, A.rows, A.cols)
    END IF
    
    ' Apply tanh to each element
    FOR i = 0 TO A.rows - 1
        FOR j = 0 TO A.cols - 1
            B.data(i, j) = TANH(A.data(i, j))
        NEXT j
    NEXT i
END SUB

' Apply GELU activation function element-wise
' GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
SUB MatrixGELU(A AS Matrix, BYREF B AS Matrix)
    DIM i AS INTEGER, j AS INTEGER
    DIM sqrt_2_div_pi AS SINGLE
    DIM x AS SINGLE, x3 AS SINGLE
    
    ' Constants for GELU approximation
    sqrt_2_div_pi = 0.7978845608 ' sqrt(2/π)
    
    ' Initialize result matrix if needed
    IF B.rows <> A.rows OR B.cols <> A.cols THEN
        FreeMatrix(B)
        InitMatrix(B, A.rows, A.cols)
    END IF
    
    ' Apply GELU to each element
    FOR i = 0 TO A.rows - 1
        FOR j = 0 TO A.cols - 1
            x = A.data(i, j)
            x3 = x * x * x
            B.data(i, j) = 0.5 * x * (1 + TANH(sqrt_2_div_pi * (x + 0.044715 * x3)))
        NEXT j
    NEXT i
END SUB

' Apply layer normalization to each row of a matrix
' Y = (X - mean) / sqrt(variance + epsilon) * gamma + beta
SUB MatrixLayerNorm(A AS Matrix, gamma AS Matrix, beta AS Matrix, epsilon AS SINGLE, BYREF B AS Matrix)
    DIM i AS INTEGER, j AS INTEGER
    DIM row_mean AS SINGLE, row_var AS SINGLE
    
    ' Check dimensions
    IF gamma.cols <> A.cols OR beta.cols <> A.cols THEN
        PRINT "ERROR: Gamma and beta dimensions incompatible with input"
        RETURN
    END IF
    
    ' Initialize result matrix if needed
    IF B.rows <> A.rows OR B.cols <> A.cols THEN
        FreeMatrix(B)
        InitMatrix(B, A.rows, A.cols)
    END IF
    
    ' Process each row
    FOR i = 0 TO A.rows - 1
        ' Compute mean
        row_mean = 0.0
        FOR j = 0 TO A.cols - 1
            row_mean = row_mean + A.data(i, j)
        NEXT j
        row_mean = row_mean / A.cols
        
        ' Compute variance
        row_var = 0.0
        FOR j = 0 TO A.cols - 1
            row_var = row_var + (A.data(i, j) - row_mean) * (A.data(i, j) - row_mean)
        NEXT j
        row_var = row_var / A.cols
        
        ' Normalize and scale
        FOR j = 0 TO A.cols - 1
            B.data(i, j) = ((A.data(i, j) - row_mean) / SQR(row_var + epsilon)) * gamma.data(0, j) + beta.data(0, j)
        NEXT j
    NEXT i
END SUB

' *******************************************************
' * Testing Functions                                   *
' *******************************************************

' Test basic matrix operations
SUB TestBasicMatrixOps()
    DIM a AS Matrix, b AS Matrix, c AS Matrix
    DIM i AS INTEGER, j AS INTEGER
    
    PRINT "Testing basic matrix operations..."
    
    ' Initialize test matrices
    InitMatrix(a, 3, 4)
    InitMatrix(b, 4, 2)
    
    ' Fill matrices with test data
    FOR i = 0 TO a.rows - 1
        FOR j = 0 TO a.cols - 1
            a.data(i, j) = i + j * 0.1
        NEXT j
    NEXT i
    
    FOR i = 0 TO b.rows - 1
        FOR j = 0 TO b.cols - 1
            b.data(i, j) = i * 0.5 - j * 0.2
        NEXT j
    NEXT i
    
    ' Test matrix multiplication
    PRINT "Matrix A:"
    PrintMatrix(a, "A")
    
    PRINT "Matrix B:"
    PrintMatrix(b, "B")
    
    PRINT "Testing matrix multiplication (A * B)..."
    MatrixMultiply(a, b, c)
    PrintMatrix(c, "A * B")
    
    ' Test transpose operation
    PRINT "Testing matrix transposition (A^T)..."
    MatrixTranspose(a, c)
    PrintMatrix(c, "A^T")
    
    ' Test matrix-matrix multiplication with transposed matrix
    PRINT "Testing multiplication with transposed B (A * B^T)..."
    DIM b_trans AS Matrix
    MatrixTranspose(b, b_trans)
    MatrixMultiply(a, b_trans, c)
    PrintMatrix(c, "A * B^T")
    
    ' Compare with direct transpose multiplication
    PRINT "Comparing with MatrixMultiplyTransposeB..."
    MatrixMultiplyTransposeB(a, b, c)
    PrintMatrix(c, "A * B^T (direct)")
    
    ' Test scaling
    PRINT "Testing matrix scaling (2 * A)..."
    MatrixScale(a, 2.0, c)
    PrintMatrix(c, "2 * A")
    
    ' Free matrices
    FreeMatrix(a)
    FreeMatrix(b)
    FreeMatrix(c)
    FreeMatrix(b_trans)
    
    PRINT "Basic matrix operations test completed."
END SUB

' Test SIMD-optimized matrix operations
SUB TestSIMDMatrixOps()
    DIM a AS Matrix, b AS Matrix, c1 AS Matrix, c2 AS Matrix
    DIM i AS INTEGER, j AS INTEGER
    DIM start_time AS DOUBLE, end_time AS DOUBLE
    DIM std_time AS DOUBLE, simd_time AS DOUBLE
    
    PRINT "Testing SIMD-optimized matrix operations..."
    
    ' Initialize test matrices (larger for better performance comparison)
    InitMatrix(a, 32, 32)
    InitMatrix(b, 32, 32)
    
    ' Fill matrices with test data
    FOR i = 0 TO a.rows - 1
        FOR j = 0 TO a.cols - 1
            a.data(i, j) = (i * j) / (a.rows * a.cols) * 2 - 1 ' Range: -1 to 1
        NEXT j
    NEXT i
    
    FOR i = 0 TO b.rows - 1
        FOR j = 0 TO b.cols - 1
            b.data(i, j) = (i + j) / (b.rows + b.cols) * 2 - 1 ' Range: -1 to 1
        NEXT j
    NEXT i
    
    ' Compare standard and SIMD matrix multiplication
    PRINT "Comparing standard and SIMD matrix multiplication..."
    
    ' Standard multiplication
    start_time = TIMER
    MatrixMultiply(a, b, c1)
    end_time = TIMER
    std_time = end_time - start_time
    
    ' SIMD multiplication
    start_time = TIMER
    MatrixMultiplySIMD(a, b, c2)
    end_time = TIMER
    simd_time = end_time - start_time
    
    ' Calculate max difference
    DIM max_diff AS SINGLE
    max_diff = 0.0
    
    FOR i = 0 TO c1.rows - 1
        FOR j = 0 TO c1.cols - 1
            DIM diff AS SINGLE
            diff = ABS(c1.data(i, j) - c2.data(i, j))
            IF diff > max_diff THEN max_diff = diff
        NEXT j
    NEXT i
    
    PRINT "Standard multiplication time: "; std_time; " seconds"
    PRINT "SIMD multiplication time    : "; simd_time; " seconds"
    PRINT "Speedup                     : "; std_time / simd_time; "x"
    PRINT "Maximum error               : "; max_diff
    
    ' Free matrices
    FreeMatrix(a)
    FreeMatrix(b)
    FreeMatrix(c1)
    FreeMatrix(c2)
    
    PRINT "SIMD matrix operations test completed."
END SUB

' Test advanced matrix operations
SUB TestAdvancedMatrixOps()
    DIM a AS Matrix, b AS Matrix
    DIM i AS INTEGER, j AS INTEGER
    
    PRINT "Testing advanced matrix operations..."
    
    ' Test matrix inversion
    PRINT "Testing matrix inversion..."
    InitMatrix(a, 3, 3)
    
    ' Create a non-singular matrix
    a.data(0, 0) = 4: a.data(0, 1) = 2: a.data(0, 2) = 0
    a.data(1, 0) = 2: a.data(1, 1) = 5: a.data(1, 2) = 1
    a.data(2, 0) = 0: a.data(2, 1) = 1: a.data(2, 2) = 3
    
    PrintMatrix(a, "A")
    
    ' Compute inverse
    DIM success AS INTEGER
    success = MatrixInverse(a, b)
    
    IF success THEN
        PrintMatrix(b, "A^(-1)")
        
        ' Verify by multiplying A * A^(-1)
        DIM identity AS Matrix
        MatrixMultiply(a, b, identity)
        PrintMatrix(identity, "A * A^(-1) (should be identity)")
        FreeMatrix(identity)
    ELSE
        PRINT "Inversion failed."
    END IF
    
    ' Test softmax
    PRINT "Testing softmax operation..."
    FreeMatrix(a)
    InitMatrix(a, 2, 4)
    
    a.data(0, 0) = 1.0: a.data(0, 1) = 2.0: a.data(0, 2) = 3.0: a.data(0, 3) = 4.0
    a.data(1, 0) = 0.1: a.data(1, 1) = 0.2: a.data(1, 2) = 0.3: a.data(1, 3) = 0.4
    
    PrintMatrix(a, "A")
    MatrixSoftmax(a, b)
    PrintMatrix(b, "softmax(A)")
    
    ' Verify softmax rows sum to 1
    PRINT "Verifying softmax row sums (should be 1.0):"
    FOR i = 0 TO b.rows - 1
        DIM row_sum AS SINGLE
        row_sum = 0.0
        FOR j = 0 TO b.cols - 1
            row_sum = row_sum + b.data(i, j)
        NEXT j
        PRINT "Row "; i; " sum: "; row_sum
    NEXT i
    
    ' Test layer normalization
    PRINT "Testing layer normalization..."
    FreeMatrix(a)
    InitMatrix(a, 2, 4)
    
    a.data(0, 0) = 1.0: a.data(0, 1) = 2.0: a.data(0, 2) = 3.0: a.data(0, 3) = 4.0
    a.data(1, 0) = 5.0: a.data(1, 1) = 6.0: a.data(1, 2) = 7.0: a.data(1, 3) = 8.0
    
    ' Create gamma and beta parameters
    DIM gamma AS Matrix, beta AS Matrix
    InitMatrix(gamma, 1, 4)
    InitMatrix(beta, 1, 4)
    
    FOR j = 0 TO 3
        gamma.data(0, j) = 1.0 ' Initially no scaling
        beta.data(0, j) = 0.0  ' Initially no shift
    NEXT j
    
    PrintMatrix(a, "A")
    MatrixLayerNorm(a, gamma, beta, 0.00001, b)
    PrintMatrix(b, "LayerNorm(A)")
    
    ' Free matrices
    FreeMatrix(a)
    FreeMatrix(b)
    FreeMatrix(gamma)
    FreeMatrix(beta)
    
    PRINT "Advanced matrix operations test completed."
END SUB

' Main test routine for matrix operations
SUB TestMatrixOps()
    PRINT "Testing Matrix Operations Module"
    PRINT "================================"
    PRINT
    
    ' Initialize matrix operations
    InitMatrixOps()
    
    ' Run basic tests
    TestBasicMatrixOps()
    PRINT
    
    ' Run SIMD tests
    TestSIMDMatrixOps()
    PRINT
    
    ' Run advanced tests
    TestAdvancedMatrixOps()
    PRINT
    
    ' Print statistics
    PrintMatrixStats()
END SUB
