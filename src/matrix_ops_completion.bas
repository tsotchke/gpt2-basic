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
            augmented.data(i, j) = A.data(i, j)            ' Left side: A
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
