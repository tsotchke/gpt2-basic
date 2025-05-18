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
            ' Similar to 8-bit but with 4-bit precision
            ' Implementation would be similar but with 4-bit packing
            ' For now, fall back to 8-bit implementation
            ' TODO: Implement native 4-bit version
            
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
            
            ' TODO: Implement native 16-bit matrix multiplication with SIMD
            ' For now, use non-SIMD implementation
            
            idx = 0
            FOR i = 0 TO A.rows - 1
                FOR j = 0 TO B.cols - 1
                    DIM sum AS LONG
                    sum = 0
                    FOR k = 0 TO A.cols - 1
                        sum = sum + (CLNG(A_16bit(i * A.cols + k)) * CLNG(B_16bit(k * B.cols + j))) / 32767
                    NEXT k
                    C_16bit(idx) = MAX(-32768, MIN(32767, sum))
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
        IF ABS
