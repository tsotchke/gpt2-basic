' *******************************************************
' * Block-Sparse Attention for GPT-2 BASIC               *
' *******************************************************
' * This module implements block-sparse representation   *
' * and operations for efficient attention computation   *
' * with limited memory.                                 *
' *                                                      *
' * Instead of storing full dense matrices, we divide    *
' * matrices into blocks and only store non-zero blocks, *
' * dramatically reducing memory usage for attention     *
' * with minimal accuracy loss.                          *
' *******************************************************

#INCLUDE "src/data_structures.bas"
#INCLUDE "src/matrix_ops.bas"
#INCLUDE "src/simd_ops.bas"

' *******************************************************
' * Constants and Type Definitions                      *
' *******************************************************

' Block-sparse storage type for matrices
TYPE SparseBlock
    row_start AS INTEGER       ' Starting row of this block
    col_start AS INTEGER       ' Starting column of this block
    block_size AS INTEGER      ' Size of this block (typically 32 or 64)
    data AS Matrix             ' The actual data for this block
    next_block AS SparseBlock PTR ' Pointer to next block in the list
END TYPE

' A sparse matrix is represented as a list of blocks
TYPE SparseBlockMatrix
    rows AS INTEGER            ' Total rows in the full matrix
    cols AS INTEGER            ' Total columns in the full matrix
    block_size AS INTEGER      ' Size of blocks (typically 32 or 64)
    num_blocks AS INTEGER      ' Number of blocks stored
    density AS SINGLE          ' Estimated density (blocks stored / total blocks)
    first_block AS SparseBlock PTR ' Pointer to first block in the list
    last_block AS SparseBlock PTR  ' Pointer to last block for faster appends
END TYPE

' Memory tracking
DIM SHARED g_blocks_allocated AS INTEGER
DIM SHARED g_blocks_freed AS INTEGER
DIM SHARED g_total_block_memory AS LONG

' Block sparsity configuration
DIM SHARED g_default_block_size AS INTEGER = 32 ' Default size of blocks
DIM SHARED g_sparsity_threshold AS SINGLE = 0.1 ' Threshold for considering a block sparse
DIM SHARED g_min_block_density AS SINGLE = 0.3  ' Minimum density to store a block

' *******************************************************
' * Block-Sparse Matrix Creation                        *
' *******************************************************

' Initialize a new sparse block matrix
SUB InitSparseBlockMatrix(BYREF sbm AS SparseBlockMatrix, rows AS INTEGER, cols AS INTEGER, block_size AS INTEGER)
    sbm.rows = rows
    sbm.cols = cols
    sbm.block_size = block_size
    sbm.num_blocks = 0
    sbm.density = 0.0
    sbm.first_block = NULL
    sbm.last_block = NULL
END SUB

' Create a new sparse block
FUNCTION CreateSparseBlock(row_start AS INTEGER, col_start AS INTEGER, block_size AS INTEGER) AS SparseBlock PTR
    DIM block AS SparseBlock PTR
    block = NEW SparseBlock
    
    block->row_start = row_start
    block->col_start = col_start
    block->block_size = block_size
    block->next_block = NULL
    
    ' Initialize data matrix for the block
    InitMatrix(block->data, block_size, block_size)
    
    ' Track memory usage
    g_blocks_allocated = g_blocks_allocated + 1
    g_total_block_memory = g_total_block_memory + (block_size * block_size * 4) ' 4 bytes per float
    
    RETURN block
END FUNCTION

' Add a block to a sparse block matrix
SUB AddBlockToSparseMatrix(BYREF sbm AS SparseBlockMatrix, block AS SparseBlock PTR)
    ' First block case
    IF sbm.first_block = NULL THEN
        sbm.first_block = block
        sbm.last_block = block
    ELSE
        ' Append to the end
        sbm.last_block->next_block = block
        sbm.last_block = block
    END IF
    
    sbm.num_blocks = sbm.num_blocks + 1
    
    ' Update density estimate
    DIM total_blocks AS INTEGER = CEILING(sbm.rows / sbm.block_size) * CEILING(sbm.cols / sbm.block_size)
    sbm.density = sbm.num_blocks / total_blocks
END SUB

' Convert a dense matrix to block-sparse format with thresholding
' Only blocks with sufficient non-zero entries are stored
SUB DenseToSparseBlockMatrix(dense_matrix AS Matrix, BYREF sbm AS SparseBlockMatrix, block_size AS INTEGER)
    DIM i AS INTEGER, j AS INTEGER, r AS INTEGER, c AS INTEGER
    DIM block_count AS INTEGER = 0
    DIM value AS SINGLE
    DIM block_sum AS SINGLE
    DIM block_nonzeros AS INTEGER
    
    InitSparseBlockMatrix(sbm, dense_matrix.rows, dense_matrix.cols, block_size)
    
    ' Process each block
    FOR r = 0 TO dense_matrix.rows - 1 STEP block_size
        FOR c = 0 TO dense_matrix.cols - 1 STEP block_size
            ' Calculate actual block dimensions (handling edge cases)
            DIM block_rows AS INTEGER = MIN(block_size, dense_matrix.rows - r)
            DIM block_cols AS INTEGER = MIN(block_size, dense_matrix.cols - c)
            
            ' Count non-zeros and sum values in this block
            block_sum = 0.0
            block_nonzeros = 0
            
            FOR i = 0 TO block_rows - 1
                FOR j = 0 TO block_cols - 1
                    value = dense_matrix.data(r + i, c + j)
                    IF ABS(value) > g_sparsity_threshold THEN
                        block_nonzeros = block_nonzeros + 1
                    END IF
                    block_sum = block_sum + ABS(value)
                NEXT j
            NEXT i
            
            ' Decide whether to keep this block
            DIM block_density AS SINGLE = block_nonzeros / (block_rows * block_cols)
            
            IF block_density >= g_min_block_density OR block_sum > g_sparsity_threshold * 10 THEN
                ' Create and populate the sparse block
                DIM new_block AS SparseBlock PTR = CreateSparseBlock(r, c, block_size)
                
                ' Copy data into the block
                FOR i = 0 TO block_rows - 1
                    FOR j = 0 TO block_cols - 1
                        new_block->data.data(i, j) = dense_matrix.data(r + i, c + j)
                    NEXT j
                NEXT i
                
                ' Add block to the sparse matrix
                AddBlockToSparseMatrix(sbm, new_block)
                block_count = block_count + 1
            END IF
        NEXT c
    NEXT r
    
    PRINT "Converted dense matrix to sparse format: ";
    PRINT block_count; " blocks stored out of "; 
    PRINT CEILING(dense_matrix.rows / block_size) * CEILING(dense_matrix.cols / block_size);
    PRINT " possible blocks ("; 
    PRINT FORMAT(sbm.density * 100, "0.00"); "% density)."
END SUB

' Convert block-sparse matrix back to dense format
SUB SparseBlockToDenseMatrix(sparse_matrix AS SparseBlockMatrix, BYREF dense_matrix AS Matrix)
    DIM i AS INTEGER, j AS INTEGER
    DIM block AS SparseBlock PTR
    
    ' Initialize the dense matrix with zeros
    InitMatrix(dense_matrix, sparse_matrix.rows, sparse_matrix.cols)
    ZeroMatrix(dense_matrix)
    
    ' Iterate through all blocks
    block = sparse_matrix.first_block
    
    WHILE block <> NULL
        ' Copy each block's data to the appropriate place in the dense matrix
        FOR i = 0 TO block->block_size - 1
            ' Skip if we're beyond the dense matrix dimensions
            IF block->row_start + i >= dense_matrix.rows THEN
                EXIT FOR
            END IF
            
            FOR j = 0 TO block->block_size - 1
                ' Skip if we're beyond the dense matrix dimensions
                IF block->col_start + j >= dense_matrix.cols THEN
                    EXIT FOR
                END IF
                
                dense_matrix.data(block->row_start + i, block->col_start + j) = block->data.data(i, j)
            NEXT j
        NEXT i
        
        block = block->next_block
    WEND
END SUB

' Free all memory associated with a sparse block matrix
SUB FreeSparseBlockMatrix(BYREF sbm AS SparseBlockMatrix)
    DIM block AS SparseBlock PTR
    DIM next_block AS SparseBlock PTR
    
    block = sbm.first_block
    
    WHILE block <> NULL
        next_block = block->next_block
        
        ' Free the matrix data
        FreeMatrix(block->data)
        
        ' Free the block itself
        DELETE block
        g_blocks_freed = g_blocks_freed + 1
        
        block = next_block
    WEND
    
    ' Reset the sparse matrix
    sbm.first_block = NULL
    sbm.last_block = NULL
    sbm.num_blocks = 0
END SUB

' *******************************************************
' * Block-Sparse Matrix Operations                      *
' *******************************************************

' Apply causal masking to a block-sparse attention matrix
' This ensures attention only flows from earlier to current tokens
SUB ApplyCausalMaskToSparseMatrix(BYREF sbm AS SparseBlockMatrix)
    DIM block AS SparseBlock PTR
    DIM i AS INTEGER, j AS INTEGER
    
    block = sbm.first_block
    
    WHILE block <> NULL
        FOR i = 0 TO block->block_size - 1
            ' Global row position
            DIM global_row AS INTEGER = block->row_start + i
            
            FOR j = 0 TO block->block_size - 1
                ' Global column position
                DIM global_col AS INTEGER = block->col_start + j
                
                ' Apply causal masking: zero out attention to future tokens
                IF global_col > global_row THEN
                    block->data.data(i, j) = 0.0
                END IF
            NEXT j
        NEXT i
        
        block = block->next_block
    WEND
END SUB

' Perform block-sparse matrix multiplication: C = A * B
' Where A is block-sparse and B is dense
SUB MultiplyBlockSparseWithDense(sparse_a AS SparseBlockMatrix, dense_b AS Matrix, BYREF dense_c AS Matrix)
    DIM block AS SparseBlock PTR
    DIM i AS INTEGER, j AS INTEGER, k AS INTEGER
    
    ' Initialize result matrix with zeros
    InitMatrix(dense_c, sparse_a.rows, dense_b.cols)
    ZeroMatrix(dense_c)
    
    ' Temporary matrix for block multiplication results
    DIM temp_result AS Matrix
    InitMatrix(temp_result, sparse_a.block_size, dense_b.cols)
    
    ' Iterate through all blocks in A
    block = sparse_a.first_block
    
    WHILE block <> NULL
        ' Zero the temporary result
        ZeroMatrix(temp_result)
        
        ' Multiply this block by the corresponding part of B
        FOR i = 0 TO block->block_size - 1
            ' Skip if we're beyond matrix bounds
            IF block->row_start + i >= sparse_a.rows THEN
                EXIT FOR
            END IF
            
            FOR j = 0 TO dense_b.cols - 1
                DIM sum AS SINGLE = 0.0
                
                FOR k = 0 TO block->block_size - 1
                    ' Skip if we're beyond matrix bounds
                    IF block->col_start + k >= sparse_a.cols THEN
                        EXIT FOR
                    END IF
                    
                    ' A(i,k) * B(k,j)
                    sum = sum + block->data.data(i, k) * dense_b.data(block->col_start + k, j)
                NEXT k
                
                temp_result.data(i, j) = sum
            NEXT j
        NEXT i
        
        ' Add the result to the appropriate position in C
        FOR i = 0 TO block->block_size - 1
            ' Skip if we're beyond matrix bounds
            IF block->row_start + i >= dense_c.rows THEN
                EXIT FOR
            END IF
            
            FOR j = 0 TO dense_b.cols - 1
                ' Skip if we're beyond matrix bounds
                IF j >= dense_c.cols THEN
                    EXIT FOR
                END IF
                
                dense_c.data(block->row_start + i, j) = dense_c.data(block->row_start + i, j) + temp_result.data(i, j)
            NEXT j
        NEXT i
        
        block = block->next_block
    WEND
    
    ' Free temporary matrix
    FreeMatrix(temp_result)
END SUB

' Softmax function for block-sparse matrix, applied row-wise
SUB SoftmaxBlockSparse(BYREF sbm AS SparseBlockMatrix)
    DIM block AS SparseBlock PTR
    DIM i AS INTEGER, j AS INTEGER
    DIM row_max AS SINGLE
    DIM row_sum AS SINGLE
    
    ' We need to determine max values for each row across all blocks
    ' For simplicity, we'll convert to dense, apply softmax, and convert back
    
    DIM dense_matrix AS Matrix
    SparseBlockToDenseMatrix(sbm, dense_matrix)
    
    ' For each row in the dense matrix
    FOR i = 0 TO dense_matrix.rows - 1
        ' Find the maximum value in this row
        row_max = -1E+30 ' Very small number
        FOR j = 0 TO dense_matrix.cols - 1
            IF dense_matrix.data(i, j) > row_max THEN
                row_max = dense_matrix.data(i, j)
            END IF
        NEXT j
        
        ' Compute exp(x - max) for numerical stability
        row_sum = 0.0
        FOR j = 0 TO dense_matrix.cols - 1
            ' Apply a more numerically stable softmax
            dense_matrix.data(i, j) = EXP(dense_matrix.data(i, j) - row_max)
            row_sum = row_sum + dense_matrix.data(i, j)
        NEXT j
        
        ' Normalize
        FOR j = 0 TO dense_matrix.cols - 1
            dense_matrix.data(i, j) = dense_matrix.data(i, j) / row_sum
        NEXT j
    NEXT i
    
    ' Clear the old sparse matrix
    FreeSparseBlockMatrix(sbm)
    
    ' Convert back to block-sparse
    DenseToSparseBlockMatrix(dense_matrix, sbm, sbm.block_size)
    
    ' Free the dense matrix
    FreeMatrix(dense_matrix)
END SUB

' Block-sparse matrix scaled dot-product attention
' Q, K, V are dense matrices, output is dense
' This computes Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
SUB BlockSparseAttention(query AS Matrix, key AS Matrix, value AS Matrix, BYREF output AS Matrix, use_causal_mask AS INTEGER)
    DIM qk_transpose AS SparseBlockMatrix
    DIM temp_result AS Matrix
    DIM d_k AS SINGLE
    DIM i AS INTEGER, j AS INTEGER
    
    ' Compute scaling factor (1/sqrt(d_k))
    d_k = 1.0 / SQR(key.cols)
    
    ' Initialize matrices for QK^T calculation
    DIM dense_qkt AS Matrix
    InitMatrix(dense_qkt, query.rows, key.rows)
    
    ' Compute Q*K^T (dense)
    MatrixMultiplyTransposeB(query, key, dense_qkt)
    
    ' Scale by 1/sqrt(d_k)
    FOR i = 0 TO dense_qkt.rows - 1
        FOR j = 0 TO dense_qkt.cols - 1
            dense_qkt.data(i, j) = dense_qkt.data(i, j) * d_k
        NEXT j
    NEXT i
    
    ' Convert to block-sparse format
    DenseToSparseBlockMatrix(dense_qkt, qk_transpose, g_default_block_size)
    
    ' Apply causal masking if needed
    IF use_causal_mask <> 0 THEN
        ApplyCausalMaskToSparseMatrix(qk_transpose)
    END IF
    
    ' Apply softmax along rows
    SoftmaxBlockSparse(qk_transpose)
    
    ' Multiply with V using block-sparse * dense multiplication
    MultiplyBlockSparseWithDense(qk_transpose, value, output)
    
    ' Free matrices
    FreeMatrix(dense_qkt)
    FreeSparseBlockMatrix(qk_transpose)
END SUB

' *******************************************************
' * Memory Optimizations and Analysis                   *
' *******************************************************

' Calculate memory usage of a sparse block matrix
FUNCTION GetSparseMatrixMemoryUsage(sbm AS SparseBlockMatrix) AS LONG
    DIM memory_usage AS LONG
    DIM block AS SparseBlock PTR
    
    ' Memory for the SparseBlockMatrix structure itself
    memory_usage = SIZEOF(SparseBlockMatrix)
    
    ' Memory for each block
    block = sbm.first_block
    WHILE block <> NULL
        ' Block structure
        memory_usage = memory_usage + SIZEOF(SparseBlock)
        
        ' Block data
        memory_usage = memory_usage + (block->block_size * block->block_size * 4) ' 4 bytes per float
        
        block = block->next_block
    WEND
    
    RETURN memory_usage
END FUNCTION

' Calculate memory usage if stored as a dense matrix
FUNCTION GetDenseMatrixMemoryUsage(rows AS INTEGER, cols AS INTEGER) AS LONG
    RETURN (rows * cols * 4) + SIZEOF(Matrix) ' 4 bytes per float
END FUNCTION

' Calculate memory savings of sparse vs dense representation
FUNCTION CalculateMemorySavings(sbm AS SparseBlockMatrix) AS SINGLE
    DIM sparse_memory AS LONG = GetSparseMatrixMemoryUsage(sbm)
    DIM dense_memory AS LONG = GetDenseMatrixMemoryUsage(sbm.rows, sbm.cols)
    
    RETURN (1.0 - (sparse_memory / dense_memory)) * 100.0
END FUNCTION

' Determine optimal block size based on matrix pattern
FUNCTION DetermineOptimalBlockSize(dense_matrix AS Matrix) AS INTEGER
    DIM block_sizes() AS INTEGER = {16, 32, 64, 128}
    DIM best_size AS INTEGER = g_default_block_size
    DIM best_memory_usage AS LONG = &H7FFFFFFF ' Max integer
    DIM i AS INTEGER
    
    FOR i = 0 TO UBOUND(block_sizes)
        DIM test_sbm AS SparseBlockMatrix
        
        ' Convert to sparse with this block size
        DenseToSparseBlockMatrix(dense_matrix, test_sbm, block_sizes(i))
        
        ' Calculate memory usage
        DIM memory_usage AS LONG = GetSparseMatrixMemoryUsage(test_sbm)
        
        ' If better than current best, update
        IF memory_usage < best_memory_usage THEN
            best_memory_usage = memory_usage
            best_size = block_sizes(i)
        END IF
        
        ' Free the test matrix
        FreeSparseBlockMatrix(test_sbm)
    NEXT i
    
    RETURN best_size
END FUNCTION

' *******************************************************
' * Testing and Benchmarking                            *
' *******************************************************

' Generate a test matrix with a specific sparsity pattern
SUB GenerateTestMatrix(BYREF mat AS Matrix, rows AS INTEGER, cols AS INTEGER, pattern_type AS INTEGER)
    DIM i AS INTEGER, j AS INTEGER
    DIM value AS SINGLE
    
    InitMatrix(mat, rows, cols)
    
    SELECT CASE pattern_type
        CASE 0: ' Random sparse (10% non-zero)
            FOR i = 0 TO rows - 1
                FOR j = 0 TO cols - 1
                    IF RND < 0.1 THEN
                        mat.data(i, j) = RND - 0.5
                    ELSE
                        mat.data(i, j) = 0.0
                    END IF
                NEXT j
            NEXT i
            
        CASE 1: ' Block diagonal
            FOR i = 0 TO rows - 1
                FOR j = 0 TO cols - 1
                    ' Elements near the diagonal
                    IF ABS(i - j) < rows \ 10 THEN
                        mat.data(i, j) = RND - 0.5
                    ELSE
                        mat.data(i, j) = 0.0
                    END IF
                NEXT j
            NEXT i
            
        CASE 2: ' Block sparse (attention-like)
            FOR i = 0 TO rows - 1
                ' Each row attends to a few random blocks
                DIM num_blocks AS INTEGER = 3 + INT(RND * 3) ' 3-5 blocks per row
                
                FOR block = 1 TO num_blocks
                    DIM block_start AS INTEGER = INT(RND * cols)
                    DIM block_width AS INTEGER = 5 + INT(RND * 15) ' 5-20 width
                    
                    FOR j = block_start TO MIN(block_start + block_width, cols - 1)
                        mat.data(i, j) = RND
                    NEXT j
                NEXT block
            NEXT i
            
        CASE 3: ' Causal attention mask (lower triangular)
            FOR i = 0 TO rows - 1
                FOR j = 0 TO i ' Only current and previous positions
                    mat.data(i, j) = RND
                NEXT j
            NEXT i
    END SELECT
END SUB

' Test the sparse block matrix operations
SUB TestBlockSparseAttention()
    DIM i AS INTEGER, j AS INTEGER
    DIM seq_len AS INTEGER = 512
    DIM embed_dim AS INTEGER = 64
    
    PRINT "Testing Block-Sparse Attention (sequence length = "; seq_len; ", embedding = "; embed_dim; ")"
    PRINT "-----------------------------------------------------------------------"
    
    ' Create test matrices
    DIM query AS Matrix, key AS Matrix, value AS Matrix
    DIM output_dense AS Matrix, output_sparse AS Matrix
    
    InitMatrix(query, seq_len, embed_dim)
    InitMatrix(key, seq_len, embed_dim)
    InitMatrix(value, seq_len, embed_dim)
    
    ' Initialize with random values
    FOR i = 0 TO seq_len - 1
        FOR j = 0 TO embed_dim - 1
            query.data(i, j) = RND - 0.5
            key.data(i, j) = RND - 0.5
            value.data(i, j) = RND - 0.5
        NEXT j
    NEXT i
    
    ' Measure dense attention performance
    DIM start_time AS DOUBLE, end_time AS DOUBLE
    start_time = TIMER
    
    ' Perform standard dense attention
    DenseAttention(query, key, value, output_dense, 1) ' 1 = use causal mask
    
    end_time = TIMER
    DIM dense_time AS DOUBLE = end_time - start_time
    
    ' Measure sparse attention performance
    start_time = TIMER
    
    ' Perform block-sparse attention
    BlockSparseAttention(query, key, value, output_sparse, 1) ' 1 = use causal mask
    
    end_time = TIMER
    DIM sparse_time AS DOUBLE = end_time - start_time
    
    ' Calculate error between dense and sparse results
    DIM max_error AS SINGLE = 0.0
    DIM avg_error AS SINGLE = 0.0
    
    FOR i = 0 TO seq_len - 1
        FOR j = 0 TO embed_dim - 1
            DIM error AS SINGLE = ABS(output_dense.data(i, j) - output_sparse.data(i, j))
            avg_error = avg_error + error
            IF error > max_error THEN max_error = error
        NEXT j
    NEXT i
    
    avg_error = avg_error / (seq_len * embed_dim)
    
    ' Report results
    PRINT "Dense attention time   : "; dense_time; " seconds"
    PRINT "Sparse attention time  : "; sparse_time; " seconds"
    PRINT "Speedup                : "; dense_time / sparse_time; "x"
    PRINT "Average error          : "; avg_error
    PRINT "Maximum error          : "; max_error
    
    ' Compute theoretical memory usage
    DIM qk_size AS LONG = GetDenseMatrixMemoryUsage(seq_len, seq_len)
    PRINT "QK^T dense memory usage: "; qk_size / 1024; " KB"
    
    ' Convert QK^T to sparse to measure actual savings
    DIM dense_qkt AS Matrix
    InitMatrix(dense_qkt, seq_len, seq_len)
    MatrixMultiplyTransposeB(query, key, dense_qkt)
    
    ' Scale by 1/sqrt(d_k)
    DIM d_k AS SINGLE = 1.0 / SQR(embed_dim)
    FOR i = 0 TO dense_qkt.rows - 1
        FOR j = 0 TO dense_qkt.cols - 1
            dense_qkt.data(i, j) = dense_qkt.data(i, j) * d_k
        NEXT j
    NEXT i
    
    ' Apply causal masking
    FOR i = 0 TO dense_qkt.rows - 1
        FOR j = i + 1 TO dense_qkt.cols - 1
            dense_qkt.data(i, j) = 0.0
        NEXT j
    NEXT i
    
    ' Convert to block-sparse
    DIM sparse_qkt AS SparseBlockMatrix
    DenseToSparseBlockMatrix(dense_qkt, sparse_qkt, g_default_block_size)
    
    DIM sparse_size AS LONG = GetSparseMatrixMemoryUsage(sparse_qkt)
    PRINT "QK^T sparse memory     : "; sparse_size / 1024; " KB"
    PRINT "Memory reduction       : "; FORMAT((1.0 - (sparse_size / qk_size)) * 100, "0.00"); "%"
    
    ' Clean up
    FreeMatrix(query)
    FreeMatrix(key)
    FreeMatrix(value)
    FreeMatrix(output_dense)
    FreeMatrix(output_sparse)
    FreeMatrix(dense_qkt)
    FreeSparseBlockMatrix(sparse_qkt)
END SUB

' Dense attention implementation for comparison
SUB DenseAttention(query AS Matrix, key AS Matrix, value AS Matrix, BYREF output AS Matrix, use_causal_mask AS INTEGER)
    DIM qkt AS Matrix
    DIM softmax_qkt AS Matrix
    DIM d_k AS SINGLE
    DIM i AS INTEGER, j AS INTEGER
    
    ' Compute scaling factor
    d_k = 1.0 / SQR(key.cols)
    
    ' Initialize matrices
    InitMatrix(qkt, query.rows, key.rows)
    InitMatrix(softmax_qkt, query.rows, key.rows)
    
    ' Compute Q*K^T
    MatrixMultiplyTransposeB(query, key, qkt)
    
    ' Scale by 1/sqrt(d_k)
    FOR i = 0 TO qkt.rows - 1
        FOR j = 0 TO qkt.cols - 1
            qkt.data(i, j) = qkt.data(i, j) * d_k
        NEXT j
    NEXT i
    
    ' Apply causal masking if needed
    IF use_causal_mask <> 0 THEN
        FOR i = 0 TO qkt.rows - 1
            FOR j = i + 1 TO qkt.cols - 1
                qkt.data(i, j) = -1E+30 ' Very negative number
            NEXT j
        NEXT i
    END IF
    
    ' Apply softmax row-wise
    FOR i = 0 TO qkt.rows - 1
        ' Find the maximum value in this row
        DIM row_max AS SINGLE = -1E+30
        FOR j = 0 TO qkt.cols - 1
            IF qkt.data(i, j) > row_max THEN
                row_max = qkt.data(i, j)
            END IF
        NEXT j
        
        ' Compute exp(x - max) for numerical stability
        DIM row_sum AS SINGLE = 0.0
        FOR j = 0 TO qkt.cols - 1
            softmax_qkt.data(i, j) = EXP(qkt.data(i, j) - row_max)
            row_sum = row_sum + softmax_qkt.data(i, j)
        NEXT j
        
        ' Normalize
        FOR j = 0 TO qkt.cols - 1
            softmax_qkt.data(i, j) = softmax_qkt.data(i, j) / row_sum
        NEXT j
    NEXT i
    
    ' Compute softmax(QK^T) * V
    InitMatrix(output, query.rows, value.cols)
    MatrixMultiply(softmax_qkt, value, output)
    
    ' Clean up
    FreeMatrix(qkt)
    FreeMatrix(softmax_qkt)
END SUB

' Main test program
SUB TestBlockSparse_Main()
    PRINT "GPT-2 BASIC Block-Sparse Attention Test"
    PRINT "========================================"
    PRINT
    
    ' Initialize random seed
    RANDOMIZE TIMER
    
    ' Test block-sparse attention
    TestBlockSparseAttention()
    
    ' Report memory tracking
    PRINT
    PRINT "Memory Usage Statistics:"
    PRINT "------------------------"
    PRINT "Blocks allocated: "; g_blocks_allocated
    PRINT "Blocks freed    : "; g_blocks_freed
    PRINT "Block memory    : "; g_total_block_memory / 1024; " KB"
    
    ' Compare different block sizes
    PRINT
    PRINT "Block Size Optimization:"
    PRINT "------------------------"
    
    DIM test_mat AS Matrix
    DIM pattern_types(0 TO 3) AS STRING
    pattern_types(0) = "Random sparse"
    pattern_types(1) = "Block diagonal"
    pattern_types(2) = "Attention-like"
    pattern_types(3) = "Causal mask"
    
    FOR pattern = 0 TO 3
        PRINT pattern_types(pattern); " pattern:"
        
        ' Generate test matrix with this pattern
        GenerateTestMatrix(test_mat, 256, 256, pattern)
        
        ' Find optimal block size
        DIM optimal_size AS INTEGER = DetermineOptimalBlockSize(test_mat)
        
        ' Report results
        PRINT "  Optimal block size: "; optimal_size
        
        ' Convert to sparse with optimal block size
        DIM sparse_mat AS SparseBlockMatrix
        DenseToSparseBlockMatrix(test_mat, sparse_mat, optimal_size)
        
        ' Calculate memory savings
        DIM savings AS SINGLE = CalculateMemorySavings(sparse_mat)
        PRINT "  Memory savings    : "; FORMAT(savings, "0.00"); "%"
        
        ' Clean up
        FreeMatrix(test_mat)
        FreeSparseBlockMatrix(sparse_mat)
        PRINT
    NEXT pattern
END SUB
