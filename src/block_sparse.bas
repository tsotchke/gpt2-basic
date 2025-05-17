' Block-sparse attention implementation for the GPT-2-like model.
' This file provides data structures and functions for efficient
' sparse attention computation optimized for 486-era constraints.

' Include necessary files
#INCLUDE "data_structures.bas"
#INCLUDE "quantization.bas" 
#INCLUDE "matrix_ops.bas"

' =============================
' Block-Sparse Matrix Structure
' =============================

' Define a structure to represent a sparse block matrix
' Instead of storing all n×n attention values, we only store blocks that are non-zero
TYPE SparseBlock
    row_start AS INTEGER    ' Starting row index of this block
    col_start AS INTEGER    ' Starting column index of this block
    block_size AS INTEGER   ' Size of the square block (typically 8, 16, etc.)
    data() AS INTEGER       ' Block data (packed LogQuantized values)
    next AS SparseBlock PTR ' Pointer to next block in the linked list
END TYPE

' Head of a linked list of sparse blocks
TYPE SparseBlockMatrix
    blocks AS SparseBlock PTR ' Pointer to the first block
    rows AS INTEGER           ' Total rows in the full matrix
    cols AS INTEGER           ' Total columns in the full matrix
    block_size AS INTEGER     ' Standard block size used
    num_blocks AS INTEGER     ' Number of blocks in the matrix
END TYPE

' =============================
' Memory Management Functions
' =============================

' Initialize a new sparse block matrix
SUB InitSparseBlockMatrix(sbm AS SparseBlockMatrix, num_rows AS INTEGER, num_cols AS INTEGER, block_size AS INTEGER)
    sbm.rows = num_rows
    sbm.cols = num_cols
    sbm.block_size = block_size
    sbm.blocks = NULL       ' No blocks initially
    sbm.num_blocks = 0
END SUB

' Add a new block to a sparse block matrix
FUNCTION AddBlock(sbm AS SparseBlockMatrix, row_start AS INTEGER, col_start AS INTEGER) AS SparseBlock PTR
    ' Create a new block
    DIM new_block AS SparseBlock PTR
    new_block = ALLOCATE(SIZEOF(SparseBlock))
    IF new_block = NULL THEN
        PRINT "Error: Memory allocation failed for new block!"
        RETURN NULL
    END IF

    ' Initialize the block
    new_block->row_start = row_start
    new_block->col_start = col_start
    new_block->block_size = sbm.block_size
    REDIM new_block->data(sbm.block_size * sbm.block_size - 1) AS INTEGER
    new_block->next = NULL

    ' Initialize data to zero (important for sparse attention)
    DIM i AS INTEGER
    FOR i = 0 TO UBOUND(new_block->data)
        ' Zero in LogQuantized format
        new_block->data(i) = FixedToLogQuantized(0).packed_value
    NEXT i

    ' Add to the linked list (at the beginning for simplicity)
    new_block->next = sbm.blocks
    sbm.blocks = new_block
    sbm.num_blocks = sbm.num_blocks + 1

    RETURN new_block
END FUNCTION

' Free a sparse block matrix
SUB FreeSparseBlockMatrix(sbm AS SparseBlockMatrix)
    DIM current AS SparseBlock PTR = sbm.blocks
    DIM next_block AS SparseBlock PTR
    
    ' Traverse the linked list and free each block
    WHILE current <> NULL
        next_block = current->next
        ERASE current->data
        DEALLOCATE(current)
        current = next_block
    WEND
    
    ' Reset the structure
    sbm.blocks = NULL
    sbm.num_blocks = 0
END SUB

' =============================
' Block-Sparse Operations
' =============================

' Find a block at specific row and column indices
' Returns NULL if the block doesn't exist
FUNCTION FindBlock(sbm AS SparseBlockMatrix, row_start AS INTEGER, col_start AS INTEGER) AS SparseBlock PTR
    DIM current AS SparseBlock PTR = sbm.blocks
    
    WHILE current <> NULL
        IF current->row_start = row_start AND current->col_start = col_start THEN
            RETURN current
        END IF
        current = current->next
    WEND
    
    RETURN NULL ' Block not found
END FUNCTION

' Get a value from the sparse matrix
' If the block doesn't exist, returns zero (in LogQuantized format)
FUNCTION GetSparseValue(sbm AS SparseBlockMatrix, row AS INTEGER, col AS INTEGER) AS INTEGER
    ' Calculate which block this element belongs to
    DIM block_row AS INTEGER = (row \ sbm.block_size) * sbm.block_size
    DIM block_col AS INTEGER = (col \ sbm.block_size) * sbm.block_size
    
    ' Find the block
    DIM block AS SparseBlock PTR = FindBlock(sbm, block_row, block_col)
    
    ' If block doesn't exist, return zero
    IF block = NULL THEN
        RETURN FixedToLogQuantized(0).packed_value
    END IF
    
    ' Calculate relative position within the block
    DIM local_row AS INTEGER = row - block->row_start
    DIM local_col AS INTEGER = col - block->col_start
    
    ' Calculate index in the flattened data array
    DIM index AS INTEGER = (local_row * block->block_size) + local_col
    
    ' Return the value
    RETURN block->data(index)
END FUNCTION

' Set a value in the sparse matrix
' Creates a new block if needed
SUB SetSparseValue(sbm AS SparseBlockMatrix, row AS INTEGER, col AS INTEGER, value AS INTEGER)
    ' Calculate which block this element belongs to
    DIM block_row AS INTEGER = (row \ sbm.block_size) * sbm.block_size
    DIM block_col AS INTEGER = (col \ sbm.block_size) * sbm.block_size
    
    ' Find or create the block
    DIM block AS SparseBlock PTR = FindBlock(sbm, block_row, block_col)
    IF block = NULL THEN
        block = AddBlock(sbm, block_row, block_col)
    END IF
    
    ' Calculate relative position within the block
    DIM local_row AS INTEGER = row - block->row_start
    DIM local_col AS INTEGER = col - block->col_start
    
    ' Calculate index in the flattened data array
    DIM index AS INTEGER = (local_row * block->block_size) + local_col
    
    ' Set the value
    block->data(index) = value
END SUB

' =============================
' Block-Sparse Attention
' =============================

' Create a block-sparse attention pattern that enforces causal masking
' Only creates blocks where attention is actually needed (upper triangular)
SUB CreateCausalAttentionPattern(sbm AS SparseBlockMatrix)
    DIM block_row AS INTEGER
    DIM block_col AS INTEGER
    DIM num_blocks_per_dim AS INTEGER = sbm.rows \ sbm.block_size
    
    ' Create blocks only for the lower triangle (where block_col <= block_row)
    ' This enforces the causal mask (a token can only attend to itself and previous tokens)
    FOR block_row = 0 TO num_blocks_per_dim - 1
        DIM row_start AS INTEGER = block_row * sbm.block_size
        
        FOR block_col = 0 TO block_row ' Only up to the current block_row (causal mask)
            DIM col_start AS INTEGER = block_col * sbm.block_size
            
            ' Add this block - it's part of the causal mask
            AddBlock(sbm, row_start, col_start)
        NEXT block_col
    NEXT block_row
END SUB

' Calculate Query * Key^T for block-sparse attention
' This is much more efficient than dense matrix multiplication
' Only performs computation for blocks in the sparse pattern
SUB BlockSparseAttentionScores(Query AS Matrix, Key AS Matrix, Scores AS SparseBlockMatrix, NEG_INF_FP AS INTEGER)
    DIM block AS SparseBlock PTR = Scores.blocks
    DIM block_size AS INTEGER = Scores.block_size
    
    ' Process each block in the sparse matrix
    WHILE block <> NULL
        DIM row_start AS INTEGER = block->row_start
        DIM col_start AS INTEGER = block->col_start
        
        ' Process this block
        DIM r AS INTEGER, c AS INTEGER, k AS INTEGER
        
        ' For each position in the block
        FOR r = 0 TO block_size - 1
            DIM global_row AS INTEGER = row_start + r
            IF global_row >= Query.rows THEN EXIT FOR ' Boundary check
            
            FOR c = 0 TO block_size - 1
                DIM global_col AS INTEGER = col_start + c
                IF global_col >= Key.rows THEN EXIT FOR ' Boundary check
                
                ' Calculate dot product for this position
                DIM dot_product AS INTEGER = 0 ' Fixed-point accumulator
                
                FOR k = 0 TO Query.cols - 1 ' Assume Query.cols = Key.cols (head_dim)
                    DIM fp_query AS INTEGER = LogQuantizedToFixed(Query.data(global_row, k))
                    DIM fp_key AS INTEGER = LogQuantizedToFixed(Key.data(global_col, k))
                    dot_product = FixedAdd(dot_product, FixedMultiply(fp_query, fp_key))
                NEXT k
                
                ' Apply causal mask (if column > row, set to negative infinity)
                IF global_col > global_row THEN
                    dot_product = NEG_INF_FP
                END IF
                
                ' Store result in the block
                DIM index AS INTEGER = r * block_size + c
                block->data(index) = FixedToLogQuantized(dot_product).packed_value
            NEXT c
        NEXT r
        
        ' Move to the next block
        block = block->next
    WEND
END SUB

' Apply Softmax to each row of a block-sparse attention matrix
SUB BlockSparseSoftmax(Scores AS SparseBlockMatrix)
    DIM r AS INTEGER
    
    ' Process each row in the matrix
    FOR r = 0 TO Scores.rows - 1
        ' Finding max value for numerical stability
        DIM max_val AS INTEGER = -2147483647 ' Minimum INT value
        DIM c AS INTEGER
        
        ' First find the maximum value in this row (across all blocks)
        FOR c = 0 TO Scores.cols - 1
            DIM val AS INTEGER = GetSparseValue(Scores, r, c)
            DIM fp_val AS INTEGER = LogQuantizedToFixed(val)
            IF fp_val > max_val THEN max_val = fp_val
        NEXT c
        
        ' Now calculate sum of exp(x - max) for normalization
        DIM exp_sum AS INTEGER = 0
        DIM exp_values(0 TO Scores.cols - 1) AS INTEGER
        
        FOR c = 0 TO Scores.cols - 1
            DIM val AS INTEGER = GetSparseValue(Scores, r, c)
            DIM fp_val AS INTEGER = LogQuantizedToFixed(val)
            
            ' Subtract max for numerical stability
            DIM shifted AS INTEGER = FixedSubtract(fp_val, max_val)
            
            ' Calculate exp(shifted) using the FixedExp function from softmax_fixed.bas
            DIM exp_val AS INTEGER = FixedExp(shifted)
            exp_values(c) = exp_val
            exp_sum = FixedAdd(exp_sum, exp_val)
        NEXT c
        
        ' Now normalize by dividing by the sum and store back in the sparse matrix
        FOR c = 0 TO Scores.cols - 1
            ' Skip division if sum is zero (numerical underflow protection)
            DIM fp_prob AS INTEGER
            IF exp_sum > 0 THEN
                fp_prob = FixedDivide(exp_values(c), exp_sum)
            ELSE
                ' If sum is zero, distribute probability uniformly
                fp_prob = FixedDivide(FIXED_POINT_SCALE, FloatToFixed(CSNG(Scores.cols)))
            END IF
            
            ' Store back in the sparse matrix
            SetSparseValue(Scores, r, c, FixedToLogQuantized(fp_prob).packed_value)
        NEXT c
    NEXT r
END SUB

' Multiply a block-sparse matrix by a dense matrix
' Scores * Value where Scores is sparse and Value is dense
SUB BlockSparseMatrixMultiply(Scores AS SparseBlockMatrix, Value AS Matrix, Output AS Matrix)
    ' Initialize output to zeros
    DIM r AS INTEGER, c AS INTEGER, v AS INTEGER
    FOR r = 0 TO Output.rows - 1
        FOR c = 0 TO Output.cols - 1
            Output.data(r, c) = FixedToLogQuantized(0).packed_value
        NEXT c
    NEXT r
    
    ' Process each block in the sparse matrix
    DIM block AS SparseBlock PTR = Scores.blocks
    DIM block_size AS INTEGER = Scores.block_size
    
    WHILE block <> NULL
        DIM row_start AS INTEGER = block->row_start
        DIM col_start AS INTEGER = block->col_start
        
        ' Process this block
        FOR r = 0 TO block_size - 1
            DIM global_row AS INTEGER = row_start + r
            IF global_row >= Output.rows THEN EXIT FOR ' Boundary check
            
            FOR c = 0 TO block_size - 1
                DIM global_col AS INTEGER = col_start + c
                IF global_col >= Value.rows THEN EXIT FOR ' Boundary check
                
                ' Get the attention score from the block
                DIM index AS INTEGER = r * block_size + c
                DIM score_packed AS INTEGER = block->data(index)
                DIM fp_score AS INTEGER = LogQuantizedToFixed(score_packed)
                
                ' Multiply score by all values in the Value matrix for this token
                FOR v = 0 TO Value.cols - 1
                    DIM fp_value AS INTEGER = LogQuantizedToFixed(Value.data(global_col, v))
                    DIM fp_product AS INTEGER = FixedMultiply(fp_score, fp_value)
                    
                    ' Add to the output
                    DIM fp_current AS INTEGER = LogQuantizedToFixed(Output.data(global_row, v))
                    DIM fp_sum AS INTEGER = FixedAdd(fp_current, fp_product)
                    Output.data(global_row, v) = FixedToLogQuantized(fp_sum).packed_value
                NEXT v
            NEXT c
        NEXT r
        
        ' Move to the next block
        block = block->next
    WEND
END SUB

' Block-sparse version of scaled dot-product attention
' This is a more efficient version of the one in transformer_components.bas
SUB BlockSparseAttention(Query AS Matrix, Key AS Matrix, Value AS Matrix, Output AS Matrix)
    ' Initialize a sparse block matrix for attention scores
    DIM block_size AS INTEGER = 16 ' Can be tuned based on performance
    DIM Scores AS SparseBlockMatrix
    InitSparseBlockMatrix(Scores, Query.rows, Key.rows, block_size)
    
    ' Create a causal attention pattern (only populate necessary blocks)
    CreateCausalAttentionPattern(Scores)
    
    ' Pre-calculate attention scaling factor in fixed-point
    ' Scale is 1 / sqrt(head_dim)
    DIM head_dim AS INTEGER = Query.cols ' Assuming Query.cols is the embedding dimension
    DIM fp_scale AS INTEGER = FixedDivide(FloatToFixed(1.0), FixedSqrt(FloatToFixed(CSNG(head_dim))))
    
    ' Define a very large negative fixed-point value for attention masking
    DIM NEG_INF_FP AS INTEGER = FloatToFixed(-1.0e9)
    
    ' Calculate attention scores (Query * Key^T) using sparse computation
    BlockSparseAttentionScores(Query, Key, Scores, NEG_INF_FP)
    
    ' Apply scaling to each block
    DIM block AS SparseBlock PTR = Scores.blocks
    WHILE block <> NULL
        DIM i AS INTEGER
        FOR i = 0 TO UBOUND(block->data)
            DIM fp_score AS INTEGER = LogQuantizedToFixed(block->data(i))
            DIM fp_scaled AS INTEGER = FixedMultiply(fp_score, fp_scale)
            block->data(i) = FixedToLogQuantized(fp_scaled).packed_value
        NEXT i
        block = block->next
    WEND
    
    ' Apply Softmax row-wise to get attention probabilities
    BlockSparseSoftmax(Scores)
    
    ' Calculate Output (Scores * Value) using sparse-dense multiplication
    BlockSparseMatrixMultiply(Scores, Value, Output)
    
    ' Clean up
    FreeSparseBlockMatrix(Scores)
END SUB

' Function to help decide when to use dense vs. sparse attention
' The dense approach may be faster for small contexts or in specific cases
FUNCTION ShouldUseBlockSparseAttention(context_length AS INTEGER) AS INTEGER
    ' Use block-sparse attention for longer contexts
    ' This threshold can be tuned based on benchmarking
    CONST SPARSE_THRESHOLD AS INTEGER = 32
    
    IF context_length > SPARSE_THRESHOLD THEN
        FUNCTION = 1 ' True, use block-sparse
    ELSE
        FUNCTION = 0 ' False, use dense
    END IF
END FUNCTION
